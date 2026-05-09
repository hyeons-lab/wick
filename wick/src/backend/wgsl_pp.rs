//! WGSL preprocessor — Rust port of llama.cpp's `pre_wgsl.hpp`.
//!
//! Converts WGSL source with C-style preprocessor directives into plain
//! WGSL ready for `wgpu::ShaderModule::create`. Variants are generated
//! at *runtime* per pipeline creation, not at build time, so the same
//! source can produce different specialized kernels under different
//! macro sets (matching how llama.cpp generates `mul_mat.wgsl`'s 24
//! `#ifdef` arms).
//!
//! Directives supported:
//! - `#include "name"` — recursive (cycle-detected); `name` resolves
//!   against the in-memory include map populated at construction.
//! - `#define NAME [VALUE]` — bare or with value (empty value → "defined
//!   as empty", treated as `1` in `#if`).
//! - `#undef NAME`.
//! - `#ifdef NAME` / `#ifndef NAME`.
//! - `#if EXPR` — full expression evaluator: `+ - * / %`, shifts
//!   `<< >>`, comparisons `< <= > >= == !=`, logical `&& || !`,
//!   `defined(NAME)` and `defined NAME`. Macros expanded recursively.
//! - `#elif EXPR`, `#else`, `#endif`.
//!
//! Macros passed via the `defines` argument override in-source
//! `#define`/`#undef` — letting the caller pin a variant.
//!
//! No function-like macros (no `#define MUL(a, b) ((a) * (b))`).

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail};

type Macros = HashMap<String, String>;

/// Stateful preprocessor. Holds the in-memory include map; `preprocess`
/// is called per shader-variant emission.
pub struct Preprocessor {
    embedded: HashMap<String, &'static str>,
}

impl Preprocessor {
    pub fn new() -> Self {
        Self {
            embedded: HashMap::new(),
        }
    }

    /// Register an include source under `name`. Subsequent
    /// `#include "name"` directives resolve against this map.
    pub fn add_include(&mut self, name: impl Into<String>, source: &'static str) {
        self.embedded.insert(name.into(), source);
    }

    /// Preprocess `source`, returning the expanded WGSL. `defines`
    /// entries are `(NAME, VALUE)`; pass `""` as VALUE for boolean
    /// `#define NAME` semantics. Defines passed here override
    /// `#define`/`#undef` directives in the source.
    pub fn preprocess(&self, source: &str, defines: &[(&str, &str)]) -> Result<String> {
        let mut macros: Macros = HashMap::new();
        let mut predefined: HashSet<String> = HashSet::new();
        for (name, value) in defines {
            macros.insert((*name).to_string(), (*value).to_string());
            predefined.insert((*name).to_string());
        }
        let mut include_stack: HashSet<String> = HashSet::new();
        let mut cond: Vec<Cond> = Vec::new();
        let mut out = String::with_capacity(source.len());
        self.process(
            source,
            &mut macros,
            &predefined,
            &mut include_stack,
            &mut cond,
            &mut out,
        )?;
        if !cond.is_empty() {
            bail!("Unclosed #if directive");
        }
        Ok(out)
    }

    fn process(
        &self,
        source: &str,
        macros: &mut Macros,
        predefined: &HashSet<String>,
        include_stack: &mut HashSet<String>,
        cond: &mut Vec<Cond>,
        out: &mut String,
    ) -> Result<()> {
        for raw_line in source.lines() {
            let trimmed = raw_line.trim_start();
            if let Some(rest) = trimmed.strip_prefix('#') {
                self.handle_directive(rest, macros, predefined, include_stack, cond, out)?;
                // Emit a blank line so post-preprocessing line numbers
                // match the source for everything outside #include
                // expansion — WGSL compiler errors land on the right
                // source line.
                out.push('\n');
            } else if cond_active(cond) {
                let expanded = expand_macros(raw_line, macros)?;
                out.push_str(&expanded);
                out.push('\n');
            } else {
                // Inactive #if branch: still emit a blank line.
                out.push('\n');
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_directive(
        &self,
        body: &str,
        macros: &mut Macros,
        predefined: &HashSet<String>,
        include_stack: &mut HashSet<String>,
        cond: &mut Vec<Cond>,
        out: &mut String,
    ) -> Result<()> {
        // Strip trailing `// comment` so #if/#elif expressions and other
        // directive args parse cleanly. Only strips `//` preceded by
        // whitespace to avoid mangling URLs like `http://...` if they
        // appear in #define values.
        let body = strip_line_comment(body.trim());
        let mut iter = body.splitn(2, char::is_whitespace);
        let cmd = iter.next().unwrap_or("");
        let args = iter.next().unwrap_or("").trim();

        match cmd {
            "include" => {
                if !cond_active(cond) {
                    return Ok(());
                }
                let name = strip_quotes(args).context("malformed #include argument")?;
                self.process_include(name, macros, predefined, include_stack, cond, out)
            }
            "define" => {
                if !cond_active(cond) {
                    return Ok(());
                }
                let mut parts = args.splitn(2, char::is_whitespace);
                let name = parts.next().unwrap_or("").trim().to_string();
                if name.is_empty() {
                    bail!("#define: missing name");
                }
                if predefined.contains(&name) {
                    return Ok(());
                }
                let value = parts.next().unwrap_or("").trim().to_string();
                macros.insert(name, value);
                Ok(())
            }
            "undef" => {
                if !cond_active(cond) {
                    return Ok(());
                }
                let name = args.split_whitespace().next().unwrap_or("");
                if name.is_empty() {
                    bail!("#undef: missing name");
                }
                if predefined.contains(name) {
                    return Ok(());
                }
                macros.remove(name);
                Ok(())
            }
            "ifdef" => {
                let name = args.split_whitespace().next().unwrap_or("");
                if name.is_empty() {
                    bail!("#ifdef: missing name");
                }
                let parent = cond_active(cond);
                let value = macros.contains_key(name);
                cond.push(Cond {
                    parent_active: parent,
                    active: parent && value,
                    taken: parent && value,
                });
                Ok(())
            }
            "ifndef" => {
                let name = args.split_whitespace().next().unwrap_or("");
                if name.is_empty() {
                    bail!("#ifndef: missing name");
                }
                let parent = cond_active(cond);
                let value = !macros.contains_key(name);
                cond.push(Cond {
                    parent_active: parent,
                    active: parent && value,
                    taken: parent && value,
                });
                Ok(())
            }
            "if" => {
                let parent = cond_active(cond);
                let value = if parent {
                    eval_expr(args, macros)? != 0
                } else {
                    false
                };
                cond.push(Cond {
                    parent_active: parent,
                    active: parent && value,
                    taken: parent && value,
                });
                Ok(())
            }
            "elif" => {
                let c = cond.last_mut().context("#elif without #if")?;
                if !c.parent_active {
                    c.active = false;
                    return Ok(());
                }
                if c.taken {
                    c.active = false;
                    return Ok(());
                }
                let value = eval_expr(args, macros)? != 0;
                c.active = value;
                if value {
                    c.taken = true;
                }
                Ok(())
            }
            "else" => {
                let c = cond.last_mut().context("#else without #if")?;
                if !c.parent_active {
                    c.active = false;
                    return Ok(());
                }
                if c.taken {
                    c.active = false;
                } else {
                    c.active = true;
                    c.taken = true;
                }
                Ok(())
            }
            "endif" => {
                cond.pop().context("#endif without #if")?;
                Ok(())
            }
            _ => bail!("Unknown directive: #{cmd}"),
        }
    }

    fn process_include(
        &self,
        name: &str,
        macros: &mut Macros,
        predefined: &HashSet<String>,
        include_stack: &mut HashSet<String>,
        cond: &mut Vec<Cond>,
        out: &mut String,
    ) -> Result<()> {
        if include_stack.contains(name) {
            bail!("Recursive include: {name}");
        }
        let source = self
            .embedded
            .get(name)
            .with_context(|| format!("Included file not found: {name}"))?;
        include_stack.insert(name.to_string());
        let result = self.process(source, macros, predefined, include_stack, cond, out);
        include_stack.remove(name);
        result
    }
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
struct Cond {
    parent_active: bool,
    active: bool,
    taken: bool,
}

fn cond_active(stack: &[Cond]) -> bool {
    stack.last().is_none_or(|c| c.active)
}

fn strip_quotes(s: &str) -> Option<&str> {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() >= 2 && bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"' {
        Some(&s[1..s.len() - 1])
    } else {
        None
    }
}

/// Strip `// comment` from the end of a directive line. Only strips
/// `//` that's at the start of the slice or preceded by ASCII
/// whitespace, so `#define URL "http://..."` is left alone.
fn strip_line_comment(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'/' && bytes[i + 1] == b'/' {
            let preceded_by_ws = i == 0 || bytes[i - 1].is_ascii_whitespace();
            if preceded_by_ws {
                return s[..i].trim_end();
            }
        }
        i += 1;
    }
    s
}

// ── Macro expansion ────────────────────────────────────────────────────────

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn expand_macros(line: &str, macros: &Macros) -> Result<String> {
    let mut visiting: HashSet<String> = HashSet::new();
    expand_internal(line, macros, &mut visiting)
}

fn expand_internal(line: &str, macros: &Macros, visiting: &mut HashSet<String>) -> Result<String> {
    let mut out = String::with_capacity(line.len());
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if is_ident_char(c) && !c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && is_ident_char(chars[i]) {
                i += 1;
            }
            let token: String = chars[start..i].iter().collect();
            match macros.get(&token) {
                None => out.push_str(&token),
                Some(value) if value.is_empty() => {
                    // C semantics: `#define FOO` (empty value) expands to nothing.
                }
                Some(value) => {
                    if visiting.contains(&token) {
                        bail!("Recursive macro: {token}");
                    }
                    visiting.insert(token.clone());
                    let expanded = expand_internal(value, macros, visiting)?;
                    visiting.remove(&token);
                    out.push_str(&expanded);
                }
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    Ok(out)
}

// ── Expression evaluator ───────────────────────────────────────────────────

fn eval_expr(expr: &str, macros: &Macros) -> Result<i64> {
    let tokens = lex_expr(expr)?;
    let mut visiting: HashSet<String> = HashSet::new();
    let mut parser = ExprParser::new(&tokens, macros, &mut visiting);
    let value = parser.parse_logical_or()?;
    parser.expect_end()?;
    Ok(value)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(i64),
    Ident(String),
    Op(&'static str),
    LParen,
    RParen,
}

fn lex_expr(input: &str) -> Result<Vec<Tok>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let s: String = chars[start..i].iter().collect();
            let n: i64 = s
                .parse()
                .map_err(|_| anyhow!("invalid number in expression: {s}"))?;
            tokens.push(Tok::Num(n));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && is_ident_char(chars[i]) {
                i += 1;
            }
            tokens.push(Tok::Ident(chars[start..i].iter().collect()));
        } else if c == '(' {
            tokens.push(Tok::LParen);
            i += 1;
        } else if c == ')' {
            tokens.push(Tok::RParen);
            i += 1;
        } else {
            // Two-char ops first — match on the char pair to avoid
            // allocating a String per non-ident character.
            let next = chars.get(i + 1).copied();
            let two_op = match (c, next) {
                ('=', Some('=')) => Some("=="),
                ('!', Some('=')) => Some("!="),
                ('<', Some('=')) => Some("<="),
                ('>', Some('=')) => Some(">="),
                ('&', Some('&')) => Some("&&"),
                ('|', Some('|')) => Some("||"),
                ('<', Some('<')) => Some("<<"),
                ('>', Some('>')) => Some(">>"),
                _ => None,
            };
            if let Some(op) = two_op {
                tokens.push(Tok::Op(op));
                i += 2;
                continue;
            }
            // Single-char ops.
            let single = match c {
                '+' => Some("+"),
                '-' => Some("-"),
                '*' => Some("*"),
                '/' => Some("/"),
                '%' => Some("%"),
                '<' => Some("<"),
                '>' => Some(">"),
                '!' => Some("!"),
                _ => None,
            };
            match single {
                Some(op) => {
                    tokens.push(Tok::Op(op));
                    i += 1;
                }
                None => bail!("unexpected character in expression: {c:?}"),
            }
        }
    }
    Ok(tokens)
}

struct ExprParser<'a> {
    tokens: &'a [Tok],
    pos: usize,
    macros: &'a Macros,
    visiting: &'a mut HashSet<String>,
}

impl<'a> ExprParser<'a> {
    fn new(tokens: &'a [Tok], macros: &'a Macros, visiting: &'a mut HashSet<String>) -> Self {
        Self {
            tokens,
            pos: 0,
            macros,
            visiting,
        }
    }

    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn accept_op(&mut self, op: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Op(o)) if *o == op) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect_end(&self) -> Result<()> {
        if self.pos != self.tokens.len() {
            bail!("trailing tokens in expression");
        }
        Ok(())
    }

    fn parse_logical_or(&mut self) -> Result<i64> {
        let mut v = self.parse_logical_and()?;
        while self.accept_op("||") {
            let rhs = self.parse_logical_and()?;
            v = i64::from((v != 0) || (rhs != 0));
        }
        Ok(v)
    }

    fn parse_logical_and(&mut self) -> Result<i64> {
        let mut v = self.parse_equality()?;
        while self.accept_op("&&") {
            let rhs = self.parse_equality()?;
            v = i64::from((v != 0) && (rhs != 0));
        }
        Ok(v)
    }

    fn parse_equality(&mut self) -> Result<i64> {
        let mut v = self.parse_relational()?;
        loop {
            if self.accept_op("==") {
                let rhs = self.parse_relational()?;
                v = i64::from(v == rhs);
            } else if self.accept_op("!=") {
                let rhs = self.parse_relational()?;
                v = i64::from(v != rhs);
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn parse_relational(&mut self) -> Result<i64> {
        let mut v = self.parse_shift()?;
        loop {
            // Check `<=` / `>=` before single-char `<` / `>`.
            if self.accept_op("<=") {
                let rhs = self.parse_shift()?;
                v = i64::from(v <= rhs);
            } else if self.accept_op(">=") {
                let rhs = self.parse_shift()?;
                v = i64::from(v >= rhs);
            } else if self.accept_op("<") {
                let rhs = self.parse_shift()?;
                v = i64::from(v < rhs);
            } else if self.accept_op(">") {
                let rhs = self.parse_shift()?;
                v = i64::from(v > rhs);
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn parse_shift(&mut self) -> Result<i64> {
        let mut v = self.parse_add()?;
        loop {
            if self.accept_op("<<") {
                let rhs = self.parse_add()?;
                v = v.wrapping_shl(rhs as u32);
            } else if self.accept_op(">>") {
                let rhs = self.parse_add()?;
                v = v.wrapping_shr(rhs as u32);
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn parse_add(&mut self) -> Result<i64> {
        let mut v = self.parse_mul()?;
        loop {
            if self.accept_op("+") {
                let rhs = self.parse_mul()?;
                v = v.wrapping_add(rhs);
            } else if self.accept_op("-") {
                let rhs = self.parse_mul()?;
                v = v.wrapping_sub(rhs);
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn parse_mul(&mut self) -> Result<i64> {
        let mut v = self.parse_unary()?;
        loop {
            if self.accept_op("*") {
                let rhs = self.parse_unary()?;
                v = v.wrapping_mul(rhs);
            } else if self.accept_op("/") {
                let rhs = self.parse_unary()?;
                v = if rhs == 0 { 0 } else { v / rhs };
            } else if self.accept_op("%") {
                let rhs = self.parse_unary()?;
                v = if rhs == 0 { 0 } else { v % rhs };
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn parse_unary(&mut self) -> Result<i64> {
        if self.accept_op("!") {
            let v = self.parse_unary()?;
            Ok(i64::from(v == 0))
        } else if self.accept_op("-") {
            let v = self.parse_unary()?;
            Ok(v.wrapping_neg())
        } else if self.accept_op("+") {
            self.parse_unary()
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<i64> {
        match self.peek().cloned() {
            Some(Tok::LParen) => {
                self.advance();
                let v = self.parse_logical_or()?;
                match self.peek() {
                    Some(Tok::RParen) => {
                        self.advance();
                        Ok(v)
                    }
                    _ => bail!("missing ')'"),
                }
            }
            Some(Tok::Num(n)) => {
                self.advance();
                Ok(n)
            }
            Some(Tok::Ident(name)) => {
                self.advance();
                if name == "defined" {
                    self.parse_defined()
                } else {
                    self.eval_ident(&name)
                }
            }
            _ => bail!("unexpected token in expression"),
        }
    }

    fn parse_defined(&mut self) -> Result<i64> {
        // Either `defined(NAME)` or `defined NAME`.
        let saw_lparen = matches!(self.peek(), Some(Tok::LParen));
        if saw_lparen {
            self.advance();
        }
        let name = match self.peek().cloned() {
            Some(Tok::Ident(n)) => {
                self.advance();
                n
            }
            _ => bail!("expected identifier after `defined`"),
        };
        if saw_lparen {
            match self.peek() {
                Some(Tok::RParen) => self.advance(),
                _ => bail!("missing ')' in defined()"),
            }
        }
        Ok(i64::from(self.macros.contains_key(&name)))
    }

    fn eval_ident(&mut self, name: &str) -> Result<i64> {
        match self.macros.get(name) {
            None => Ok(0),
            Some(value) if value.is_empty() => Ok(1),
            Some(value) => {
                if self.visiting.contains(name) {
                    bail!("Recursive macro in expression: {name}");
                }
                self.visiting.insert(name.to_string());
                let result = eval_expr_with(value, self.macros, self.visiting);
                self.visiting.remove(name);
                result
            }
        }
    }
}

fn eval_expr_with(expr: &str, macros: &Macros, visiting: &mut HashSet<String>) -> Result<i64> {
    let tokens = lex_expr(expr)?;
    let mut parser = ExprParser::new(&tokens, macros, visiting);
    let v = parser.parse_logical_or()?;
    parser.expect_end()?;
    Ok(v)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pp() -> Preprocessor {
        Preprocessor::new()
    }

    #[test]
    fn passthrough_no_directives() {
        let out = pp().preprocess("fn foo() {}\n", &[]).unwrap();
        assert_eq!(out, "fn foo() {}\n");
    }

    #[test]
    fn ifdef_taken_when_defined() {
        let src = "#ifdef VEC\nvec_path();\n#endif\n";
        let out = pp().preprocess(src, &[("VEC", "")]).unwrap();
        assert!(out.contains("vec_path()"), "got: {out:?}");
    }

    #[test]
    fn ifdef_skipped_when_not_defined() {
        let src = "#ifdef VEC\nvec_path();\n#endif\n";
        let out = pp().preprocess(src, &[]).unwrap();
        assert!(!out.contains("vec_path()"), "got: {out:?}");
    }

    #[test]
    fn ifndef_inverse_of_ifdef() {
        let src = "#ifndef VEC\nfallback();\n#endif\n";
        let with = pp().preprocess(src, &[("VEC", "")]).unwrap();
        let without = pp().preprocess(src, &[]).unwrap();
        assert!(!with.contains("fallback()"));
        assert!(without.contains("fallback()"));
    }

    #[test]
    fn if_defined_function_form() {
        let src = "#if defined(X) && defined(Y)\nboth();\n#endif\n";
        assert!(
            pp().preprocess(src, &[("X", ""), ("Y", "")])
                .unwrap()
                .contains("both()")
        );
        assert!(
            !pp()
                .preprocess(src, &[("X", "")])
                .unwrap()
                .contains("both()")
        );
    }

    #[test]
    fn if_defined_bare_form() {
        let src = "#if defined X\nyes();\n#endif\n";
        assert!(
            pp().preprocess(src, &[("X", "")])
                .unwrap()
                .contains("yes()")
        );
        assert!(!pp().preprocess(src, &[]).unwrap().contains("yes()"));
    }

    #[test]
    fn if_arithmetic_precedence() {
        // `1 + 2 * 3` == 7 (true), `1 == 0` (false).
        let src = "#if 1 + 2 * 3 == 7\ngood();\n#endif\n#if 1 == 0\nbad();\n#endif\n";
        let out = pp().preprocess(src, &[]).unwrap();
        assert!(out.contains("good()"));
        assert!(!out.contains("bad()"));
    }

    #[test]
    fn if_parens_override_precedence() {
        let src = "#if (1 + 2) * 3 == 9\nyes();\n#endif\n";
        assert!(pp().preprocess(src, &[]).unwrap().contains("yes()"));
    }

    #[test]
    fn if_logical_and_or() {
        let src = "#if 1 == 1 && 2 != 3 || 0\nyes();\n#endif\n";
        assert!(pp().preprocess(src, &[]).unwrap().contains("yes()"));
    }

    #[test]
    fn if_macro_with_value() {
        // `#if TILE_M >= 4` — TILE_M's value 4 evaluates as int.
        let src = "#if TILE_M >= 4\nbig();\n#endif\n";
        assert!(
            pp().preprocess(src, &[("TILE_M", "4")])
                .unwrap()
                .contains("big()")
        );
        assert!(
            !pp()
                .preprocess(src, &[("TILE_M", "2")])
                .unwrap()
                .contains("big()")
        );
    }

    #[test]
    fn elif_chain_picks_first_match() {
        let src =
            "#if 0\nfirst();\n#elif 1\nsecond();\n#elif 1\nthird();\n#else\nelse_();\n#endif\n";
        let out = pp().preprocess(src, &[]).unwrap();
        assert!(out.contains("second()"));
        assert!(!out.contains("first()"));
        assert!(!out.contains("third()"));
        assert!(!out.contains("else_()"));
    }

    #[test]
    fn else_taken_when_all_arms_false() {
        let src = "#if 0\na();\n#elif 0\nb();\n#else\nc();\n#endif\n";
        assert!(pp().preprocess(src, &[]).unwrap().contains("c()"));
    }

    #[test]
    fn nested_conditionals() {
        // Inner #ifdef Y inside outer #ifdef X — inner only takes
        // when both X and Y defined.
        let src = "#ifdef X\n#ifdef Y\nboth();\n#endif\nonly_x();\n#endif\n";
        let xy = pp().preprocess(src, &[("X", ""), ("Y", "")]).unwrap();
        let x_only = pp().preprocess(src, &[("X", "")]).unwrap();
        let neither = pp().preprocess(src, &[]).unwrap();
        assert!(xy.contains("both()") && xy.contains("only_x()"));
        assert!(!x_only.contains("both()") && x_only.contains("only_x()"));
        assert!(!neither.contains("only_x()") && !neither.contains("both()"));
    }

    #[test]
    fn define_and_use_in_body() {
        let src = "#define FOO bar\nfn FOO() {}\n";
        let out = pp().preprocess(src, &[]).unwrap();
        assert!(out.contains("fn bar()"), "got: {out:?}");
    }

    #[test]
    fn undef_after_define() {
        let src = "#define FOO bar\nbefore: FOO\n#undef FOO\nafter: FOO\n";
        let out = pp().preprocess(src, &[]).unwrap();
        assert!(out.contains("before: bar"), "got: {out:?}");
        assert!(out.contains("after: FOO"), "got: {out:?}");
    }

    #[test]
    fn predefined_overrides_source_define() {
        // Caller passes FOO=baz; in-source `#define FOO bar` should be
        // ignored.
        let src = "#define FOO bar\nFOO\n";
        let out = pp().preprocess(src, &[("FOO", "baz")]).unwrap();
        assert!(out.contains("baz") && !out.contains("bar"));
    }

    #[test]
    fn predefined_undef_ignored() {
        // Caller passes FOO=v; source `#undef FOO` should be ignored.
        let src = "#undef FOO\nFOO\n";
        let out = pp().preprocess(src, &[("FOO", "v")]).unwrap();
        assert!(out.contains('v'));
    }

    #[test]
    fn include_basic() {
        let mut p = pp();
        p.add_include("decls.tmpl", "fn helper() {}\n");
        let src = "#include \"decls.tmpl\"\nfn main() { helper(); }\n";
        let out = p.preprocess(src, &[]).unwrap();
        assert!(out.contains("fn helper()") && out.contains("fn main()"));
    }

    #[test]
    fn include_recursive() {
        let mut p = pp();
        p.add_include("a.tmpl", "fn a() {}\n#include \"b.tmpl\"\n");
        p.add_include("b.tmpl", "fn b() {}\n");
        let out = p.preprocess("#include \"a.tmpl\"\n", &[]).unwrap();
        assert!(out.contains("fn a()") && out.contains("fn b()"));
    }

    #[test]
    fn include_cycle_detected() {
        let mut p = pp();
        p.add_include("a.tmpl", "#include \"b.tmpl\"\n");
        p.add_include("b.tmpl", "#include \"a.tmpl\"\n");
        let err = p.preprocess("#include \"a.tmpl\"\n", &[]).unwrap_err();
        assert!(format!("{err}").contains("Recursive include"));
    }

    #[test]
    fn include_missing_errors() {
        let err = pp()
            .preprocess("#include \"nope.tmpl\"\n", &[])
            .unwrap_err();
        assert!(format!("{err}").contains("not found"));
    }

    #[test]
    fn macro_expansion_recursive_detected() {
        let src = "#define A B\n#define B A\nA\n";
        let err = pp().preprocess(src, &[]).unwrap_err();
        assert!(format!("{err}").contains("Recursive macro"));
    }

    #[test]
    fn macro_expansion_non_ident_chars_passthrough() {
        let src = "#define X 1\nlet a: i32 = X + 2;\n";
        let out = pp().preprocess(src, &[]).unwrap();
        assert!(out.contains("let a: i32 = 1 + 2;"), "got: {out:?}");
    }

    #[test]
    fn unclosed_if_errors() {
        let err = pp().preprocess("#if 1\nbody();\n", &[]).unwrap_err();
        assert!(format!("{err}").contains("Unclosed"));
    }

    #[test]
    fn endif_without_if_errors() {
        let err = pp().preprocess("#endif\n", &[]).unwrap_err();
        assert!(format!("{err}").contains("without #if"));
    }

    #[test]
    fn realistic_vec_scalar_split() {
        // Mirror the pattern in llama.cpp's mul_mat_subgroup_matrix.wgsl
        // and confirm only the requested arm survives.
        let src = "\
#ifdef VEC
fn store_dst(idx: u32) -> vec4<f32> { return vec4<f32>(0.0); }
#endif

#ifdef SCALAR
fn store_dst(idx: u32) -> f32 { return 0.0; }
#endif
";
        let vec = pp().preprocess(src, &[("VEC", "")]).unwrap();
        let scalar = pp().preprocess(src, &[("SCALAR", "")]).unwrap();
        assert!(vec.contains("vec4<f32>"));
        assert!(!vec.contains("-> f32 {"));
        assert!(scalar.contains("-> f32 {"));
        assert!(!scalar.contains("vec4<f32>"));
    }

    // ── Expression evaluator unit tests ────────────────────────────────────

    #[test]
    fn eval_basic_arithmetic() {
        let m = Macros::new();
        assert_eq!(eval_expr("1 + 2 * 3", &m).unwrap(), 7);
        assert_eq!(eval_expr("(1 + 2) * 3", &m).unwrap(), 9);
        assert_eq!(eval_expr("10 / 3", &m).unwrap(), 3);
        assert_eq!(eval_expr("10 % 3", &m).unwrap(), 1);
    }

    #[test]
    fn eval_div_by_zero_returns_zero() {
        let m = Macros::new();
        // Mirror pre_wgsl.hpp: x / 0 → 0 instead of trapping.
        assert_eq!(eval_expr("5 / 0", &m).unwrap(), 0);
        assert_eq!(eval_expr("5 % 0", &m).unwrap(), 0);
    }

    #[test]
    fn eval_logical_short_circuit_form() {
        let m = Macros::new();
        // Both arms get evaluated (no short-circuit), but result still
        // matches Boolean semantics.
        assert_eq!(eval_expr("0 && 1", &m).unwrap(), 0);
        assert_eq!(eval_expr("1 && 1", &m).unwrap(), 1);
        assert_eq!(eval_expr("0 || 0", &m).unwrap(), 0);
        assert_eq!(eval_expr("0 || 1", &m).unwrap(), 1);
    }

    #[test]
    fn eval_comparison_ops() {
        let m = Macros::new();
        assert_eq!(eval_expr("3 < 4", &m).unwrap(), 1);
        assert_eq!(eval_expr("3 <= 3", &m).unwrap(), 1);
        assert_eq!(eval_expr("3 > 3", &m).unwrap(), 0);
        assert_eq!(eval_expr("3 >= 3", &m).unwrap(), 1);
        assert_eq!(eval_expr("3 == 3", &m).unwrap(), 1);
        assert_eq!(eval_expr("3 != 3", &m).unwrap(), 0);
    }

    #[test]
    fn eval_unary_ops() {
        let m = Macros::new();
        assert_eq!(eval_expr("!0", &m).unwrap(), 1);
        assert_eq!(eval_expr("!1", &m).unwrap(), 0);
        assert_eq!(eval_expr("-5", &m).unwrap(), -5);
        assert_eq!(eval_expr("+5", &m).unwrap(), 5);
        assert_eq!(eval_expr("!!1", &m).unwrap(), 1);
    }

    #[test]
    fn eval_shifts() {
        let m = Macros::new();
        assert_eq!(eval_expr("1 << 4", &m).unwrap(), 16);
        assert_eq!(eval_expr("64 >> 2", &m).unwrap(), 16);
    }

    #[test]
    fn eval_defined_function_and_bare() {
        let mut m = Macros::new();
        m.insert("X".into(), "".into());
        assert_eq!(eval_expr("defined(X)", &m).unwrap(), 1);
        assert_eq!(eval_expr("defined X", &m).unwrap(), 1);
        assert_eq!(eval_expr("defined(Y)", &m).unwrap(), 0);
        assert_eq!(eval_expr("!defined(Y)", &m).unwrap(), 1);
    }

    #[test]
    fn eval_ident_as_int() {
        let mut m = Macros::new();
        m.insert("TILE".into(), "4".into());
        assert_eq!(eval_expr("TILE", &m).unwrap(), 4);
        assert_eq!(eval_expr("TILE * 2", &m).unwrap(), 8);
        assert_eq!(eval_expr("UNDEF", &m).unwrap(), 0);
    }

    #[test]
    fn eval_ident_empty_value_is_one() {
        let mut m = Macros::new();
        m.insert("FLAG".into(), "".into());
        assert_eq!(eval_expr("FLAG", &m).unwrap(), 1);
    }

    #[test]
    fn eval_macro_expression_recursive() {
        let mut m = Macros::new();
        m.insert("A".into(), "B + 1".into());
        m.insert("B".into(), "2".into());
        assert_eq!(eval_expr("A * 3", &m).unwrap(), (2 + 1) * 3);
    }

    #[test]
    fn eval_recursive_macro_in_expr_detected() {
        let mut m = Macros::new();
        m.insert("A".into(), "B".into());
        m.insert("B".into(), "A".into());
        let err = eval_expr("A", &m).unwrap_err();
        assert!(format!("{err}").contains("Recursive macro"));
    }

    #[test]
    fn ifdef_missing_name_errors() {
        let err = pp()
            .preprocess("#ifdef\nbody();\n#endif\n", &[])
            .unwrap_err();
        assert!(format!("{err}").contains("#ifdef: missing name"));
    }

    #[test]
    fn ifndef_missing_name_errors() {
        let err = pp()
            .preprocess("#ifndef\nbody();\n#endif\n", &[])
            .unwrap_err();
        assert!(format!("{err}").contains("#ifndef: missing name"));
    }

    #[test]
    fn empty_define_expands_to_empty_in_body() {
        // C-preprocessor semantics: `#define FOO` (no value) expands to
        // nothing in body code. Used in shaders to define feature flags
        // that are checked via `#ifdef FOO` but never substituted.
        let src = "#define FOO\nprefix FOO suffix\n";
        let out = pp().preprocess(src, &[]).unwrap();
        assert!(out.contains("prefix  suffix"), "got: {out:?}");
        assert!(!out.contains("FOO"), "got: {out:?}");
    }

    #[test]
    fn empty_predefined_expands_to_empty_in_body() {
        // Same as above but for caller-passed predefines.
        let src = "fn main() { let x = TAG; }\n";
        let out = pp().preprocess(src, &[("TAG", "")]).unwrap();
        assert!(out.contains("let x = ;"), "got: {out:?}");
        assert!(!out.contains("TAG"), "got: {out:?}");
    }

    #[test]
    fn directive_name_ignores_trailing_comment() {
        // Trailing `// comment` on directive lines is common in shader
        // sources. `args.split_whitespace().next()` picks just the name.
        let src = "#ifdef VEC // enable vector path\nvec_path();\n#endif\n";
        let out = pp().preprocess(src, &[("VEC", "")]).unwrap();
        assert!(out.contains("vec_path()"), "got: {out:?}");
        // Same for #ifndef and #undef.
        let src = "#define FOO 1\n#undef FOO // turn off\nFOO\n";
        let out = pp().preprocess(src, &[]).unwrap();
        assert!(out.contains("FOO") && !out.contains('1'), "got: {out:?}");
    }

    #[test]
    fn line_numbers_preserved_for_directives() {
        // A shader with leading directives should produce output where
        // body lines land on the same source-relative line index. WGSL
        // compiler errors then point at the correct source line.
        let src = "#ifdef VEC\n#define WIDTH 4\nfn body() {}\n#endif\n";
        let out = pp().preprocess(src, &[("VEC", "")]).unwrap();
        let lines: Vec<&str> = out.split_inclusive('\n').collect();
        // 4 input lines → 4 output lines (3 directive newlines + body).
        assert_eq!(lines.len(), 4, "got: {out:?}");
        // Body is on line 3 (1-indexed), matching its source position.
        assert!(lines[2].contains("fn body()"), "got: {out:?}");
    }

    #[test]
    fn if_strips_trailing_line_comment() {
        // `#if EXPR // comment` — comment must be stripped before
        // lex_expr, otherwise `//` tokenizes as two division operators
        // and the parser fails.
        let src = "#if VERSION >= 4 // require v4+\nyes();\n#endif\n";
        let out = pp().preprocess(src, &[("VERSION", "5")]).unwrap();
        assert!(out.contains("yes()"), "got: {out:?}");
    }

    #[test]
    fn elif_strips_trailing_line_comment() {
        let src = "#if 0\nbad();\n#elif 1 // pick this\nyes();\n#endif\n";
        let out = pp().preprocess(src, &[]).unwrap();
        assert!(out.contains("yes()"), "got: {out:?}");
        assert!(!out.contains("bad()"));
    }

    #[test]
    fn line_comment_in_url_not_stripped() {
        // `//` not preceded by whitespace (e.g. inside an URL) should
        // be left alone — this exercises the `preceded_by_ws` check.
        let s = "value//rest";
        assert_eq!(strip_line_comment(s), "value//rest");
        let s = "value // comment";
        assert_eq!(strip_line_comment(s), "value");
    }

    #[test]
    fn line_numbers_preserved_for_skipped_block() {
        // Inactive #ifdef arm still emits blank lines so the active
        // arm's lines stay on their original source-line index.
        let src = "#ifdef ABSENT\nbad();\n#else\ngood();\n#endif\n";
        let out = pp().preprocess(src, &[]).unwrap();
        let lines: Vec<&str> = out.split_inclusive('\n').collect();
        assert_eq!(lines.len(), 5, "got: {out:?}");
        // `good()` is on input line 4, so it lands on output line 4.
        assert!(lines[3].contains("good()"), "got: {out:?}");
        assert!(!out.contains("bad()"));
    }
}
