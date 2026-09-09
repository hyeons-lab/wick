// GENERATED CODE - DO NOT EDIT BY HAND
//
// Web stub for `cera_ffi`. Mirrors the public API of the generated
// bindings so that a package exporting them conditionally still COMPILES on
// targets without `dart:ffi`. Every entry point throws `UnsupportedError`.
//
// Data types (records, enums, errors) are real, not stubs: they are plain
// Dart and carry no FFI, so constructing and inspecting them works here.

library;

import 'dart:typed_data';
import 'dart:async';
import 'dart:convert';

/// Thrown by every API in this stub.
///
/// Reaching one of these means the code ran on a target without `dart:ffi`
/// (the web). The failure is deliberate and total: there is no partial
/// support to fall back to.
Never _unsupportedOnWeb(String api) {
  throw UnsupportedError(
    '$api is not available on this platform: package `cera_ffi` needs '
    'dart:ffi, which the web does not provide.',
  );
}

const _sentinel = Object();

/// PCM audio input buffer.
class AudioInput {
  const AudioInput({
    required this.pcm,
    this.sampleRate = 16000,
  });

  final List<double> pcm;
  final int sampleRate;

  Map<String, dynamic> toJson() {
    return {
      'pcm': this.pcm,
      'sampleRate': this.sampleRate,
    };
  }

  factory AudioInput.fromJson(Map<String, dynamic> json) {
    return AudioInput(
      pcm: (json['pcm'] as List).map((item) => (item as num).toDouble()).toList(),
      sampleRate: json.containsKey('sampleRate') ? (json['sampleRate'] as num).toInt() : 16000,
    );
  }

  AudioInput copyWith({
    List<double>? pcm,
    int? sampleRate,
  }) {
    return AudioInput(
      pcm: pcm ?? this.pcm,
      sampleRate: sampleRate ?? this.sampleRate,
    );
  }

  @override
  String toString() {
    return 'AudioInput(pcm: $pcm, sampleRate: $sampleRate)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is AudioInput && pcm == other.pcm && sampleRate == other.sampleRate;

  @override
  int get hashCode => Object.hash(pcm, sampleRate);
}

/// One message in a chat-template conversation. Mirrors
/// [`cera::tokenizer::ChatMessage`]. Pass a `Vec<ChatMessage>` to
/// [`CeraEngine::apply_chat_template`] to render the model's
/// chat-template (Jinja2 from GGUF metadata) into a prompt string
/// ready to feed into [`Session::append_text`].
///
/// `role` follows the OpenAI / chat-template convention — typically
/// one of `"system"`, `"user"`, `"assistant"`, occasionally
/// `"tool"`. cera-ffi doesn't validate the role string; whatever is
/// passed flows directly into the Jinja template. Whether an
/// unknown role errors or silently no-ops depends on the template's
/// own logic — many templates have an explicit error path for
/// unrecognized roles, but it's template-dependent rather than
/// enforced by [`CeraEngine::apply_chat_template`].
class ChatMessage {
  const ChatMessage({
    required this.role,
    required this.content,
  });

  final String role;
  final String content;

  Map<String, dynamic> toJson() {
    return {
      'role': this.role,
      'content': this.content,
    };
  }

  factory ChatMessage.fromJson(Map<String, dynamic> json) {
    return ChatMessage(
      role: json['role'] as String,
      content: json['content'] as String,
    );
  }

  ChatMessage copyWith({
    String? role,
    String? content,
  }) {
    return ChatMessage(
      role: role ?? this.role,
      content: content ?? this.content,
    );
  }

  @override
  String toString() {
    return 'ChatMessage(role: $role, content: $content)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ChatMessage && role == other.role && content == other.content;

  @override
  int get hashCode => Object.hash(role, content);
}

/// Per-engine configuration at load time. Mirrors [`cera::EngineConfig`]
/// with `u64` fields (UniFFI doesn't marshal `usize`).
class EngineConfig {
  const EngineConfig({
    /// KV-cache capacity in tokens. Capped by the model's own
    /// `max_seq_len`. Pass `0` to use the model's full declared
    /// `max_seq_len` (translated to `usize::MAX` internally, then
    /// capped by the loader).
    this.contextSize = 4096,
    required this.backend,
    /// Bundle repository for resolving `http(s)://` URLs in manifests
    /// (or for [`CeraEngine::from_bundle_id`]). `None` means "remote
    /// URLs will fail with an error"; set this to a [`BundleRepo`]
    /// rooted at a persistent cache directory to enable remote
    /// downloads. Construct the repo once + reuse it across engine
    /// loads so its HTTP client pool + on-disk cache are shared.
    this.bundleRepo = null,
    /// Optional path to a DSpark speculative draft model GGUF file.
    this.draftModel = null,
    /// Whether to prefer GPU depthformer for audio decoder generation.
    this.gpuDepthformer = false,
  });

  /// KV-cache capacity in tokens. Capped by the model's own
  /// `max_seq_len`. Pass `0` to use the model's full declared
  /// `max_seq_len` (translated to `usize::MAX` internally, then
  /// capped by the loader).
  final int contextSize;
  final BackendPreference backend;
  /// Bundle repository for resolving `http(s)://` URLs in manifests
  /// (or for [`CeraEngine::from_bundle_id`]). `None` means "remote
  /// URLs will fail with an error"; set this to a [`BundleRepo`]
  /// rooted at a persistent cache directory to enable remote
  /// downloads. Construct the repo once + reuse it across engine
  /// loads so its HTTP client pool + on-disk cache are shared.
  final BundleRepo? bundleRepo;
  /// Optional path to a DSpark speculative draft model GGUF file.
  final String? draftModel;
  /// Whether to prefer GPU depthformer for audio decoder generation.
  final bool gpuDepthformer;

  Map<String, dynamic> toJson() {
    return {
      'contextSize': this.contextSize,
      'backend': BackendPreferenceFfiCodec.encode(this.backend),
      'bundleRepo': this.bundleRepo == null ? null : (() { final __tmp = this.bundleRepo!; return BundleRepoFfiCodec.lower(__tmp); })(),
      'draftModel': this.draftModel,
      'gpuDepthformer': this.gpuDepthformer,
    };
  }

  factory EngineConfig.fromJson(Map<String, dynamic> json) {
    return EngineConfig(
      contextSize: json.containsKey('contextSize') ? (json['contextSize'] as num).toInt() : 4096,
      backend: BackendPreferenceFfiCodec.decode(json['backend'] as String),
      bundleRepo: json.containsKey('bundleRepo') ? json['bundleRepo'] == null ? null : (() { final __tmp = json['bundleRepo']; return BundleRepoFfiCodec.lift((__tmp as num).toInt()); })() : null,
      draftModel: json.containsKey('draftModel') ? json['draftModel'] == null ? null : json['draftModel'] as String : null,
      gpuDepthformer: json.containsKey('gpuDepthformer') ? json['gpuDepthformer'] as bool : false,
    );
  }

  EngineConfig copyWith({
    int? contextSize,
    BackendPreference? backend,
    Object? bundleRepo = _sentinel,
    Object? draftModel = _sentinel,
    bool? gpuDepthformer,
  }) {
    return EngineConfig(
      contextSize: contextSize ?? this.contextSize,
      backend: backend ?? this.backend,
      bundleRepo: bundleRepo == _sentinel ? this.bundleRepo : bundleRepo as BundleRepo?,
      draftModel: draftModel == _sentinel ? this.draftModel : draftModel as String?,
      gpuDepthformer: gpuDepthformer ?? this.gpuDepthformer,
    );
  }

  @override
  String toString() {
    return 'EngineConfig(contextSize: $contextSize, backend: $backend, bundleRepo: $bundleRepo, draftModel: $draftModel, gpuDepthformer: $gpuDepthformer)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is EngineConfig && contextSize == other.contextSize && backend == other.backend && bundleRepo == other.bundleRepo && draftModel == other.draftModel && gpuDepthformer == other.gpuDepthformer;

  @override
  int get hashCode => Object.hash(contextSize, backend, bundleRepo, draftModel, gpuDepthformer);
}

/// An identified PII entity span in source text.
class FfiEntitySpan {
  const FfiEntitySpan({
    /// Entity label type (e.g. "NAME", "EMAIL", "PHONE_NUMBER", "STREET_ADDRESS").
    required this.entityType,
    /// UTF-8 character start index in source text (inclusive).
    required this.startChar,
    /// UTF-8 character end index in source text (exclusive).
    required this.endChar,
    /// Token start index in sequence (inclusive).
    required this.startToken,
    /// Token end index in sequence (exclusive).
    required this.endToken,
    /// Extracted text slice.
    required this.text,
    /// Mean classification confidence score across the span tokens [0.0..1.0].
    required this.score,
  });

  /// Entity label type (e.g. "NAME", "EMAIL", "PHONE_NUMBER", "STREET_ADDRESS").
  final String entityType;
  /// UTF-8 character start index in source text (inclusive).
  final int startChar;
  /// UTF-8 character end index in source text (exclusive).
  final int endChar;
  /// Token start index in sequence (inclusive).
  final int startToken;
  /// Token end index in sequence (exclusive).
  final int endToken;
  /// Extracted text slice.
  final String text;
  /// Mean classification confidence score across the span tokens [0.0..1.0].
  final double score;

  Map<String, dynamic> toJson() {
    return {
      'entityType': this.entityType,
      'startChar': this.startChar,
      'endChar': this.endChar,
      'startToken': this.startToken,
      'endToken': this.endToken,
      'text': this.text,
      'score': this.score,
    };
  }

  factory FfiEntitySpan.fromJson(Map<String, dynamic> json) {
    return FfiEntitySpan(
      entityType: json['entityType'] as String,
      startChar: (json['startChar'] as num).toInt(),
      endChar: (json['endChar'] as num).toInt(),
      startToken: (json['startToken'] as num).toInt(),
      endToken: (json['endToken'] as num).toInt(),
      text: json['text'] as String,
      score: (json['score'] as num).toDouble(),
    );
  }

  FfiEntitySpan copyWith({
    String? entityType,
    int? startChar,
    int? endChar,
    int? startToken,
    int? endToken,
    String? text,
    double? score,
  }) {
    return FfiEntitySpan(
      entityType: entityType ?? this.entityType,
      startChar: startChar ?? this.startChar,
      endChar: endChar ?? this.endChar,
      startToken: startToken ?? this.startToken,
      endToken: endToken ?? this.endToken,
      text: text ?? this.text,
      score: score ?? this.score,
    );
  }

  @override
  String toString() {
    return 'FfiEntitySpan(entityType: $entityType, startChar: $startChar, endChar: $endChar, startToken: $startToken, endToken: $endToken, text: $text, score: $score)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiEntitySpan && entityType == other.entityType && startChar == other.startChar && endChar == other.endChar && startToken == other.startToken && endToken == other.endToken && text == other.text && score == other.score;

  @override
  int get hashCode => Object.hash(entityType, startChar, endChar, startToken, endToken, text, score);
}

/// Configuration options for keyword spotting.
class FfiHotwordConfig {
  const FfiHotwordConfig({
    /// Activation probability threshold (default: 0.75).
    required this.threshold,
    /// Post-detection debounce cooldown in milliseconds (default: 2000 ms).
    required this.cooldownMs,
    /// Evaluation step interval in milliseconds (default: 80 ms).
    required this.stepMs,
    /// Sliding window length in milliseconds (default: 1200 ms).
    required this.windowMs,
    /// Audio pre-roll margin in milliseconds preserved before command (default: 150 ms).
    required this.preRollMs,
    /// VAD speech probability threshold for gating KWS (default: 0.5).
    required this.vadThreshold,
  });

  /// Activation probability threshold (default: 0.75).
  final double threshold;
  /// Post-detection debounce cooldown in milliseconds (default: 2000 ms).
  final int cooldownMs;
  /// Evaluation step interval in milliseconds (default: 80 ms).
  final int stepMs;
  /// Sliding window length in milliseconds (default: 1200 ms).
  final int windowMs;
  /// Audio pre-roll margin in milliseconds preserved before command (default: 150 ms).
  final int preRollMs;
  /// VAD speech probability threshold for gating KWS (default: 0.5).
  final double vadThreshold;

  Map<String, dynamic> toJson() {
    return {
      'threshold': this.threshold,
      'cooldownMs': this.cooldownMs,
      'stepMs': this.stepMs,
      'windowMs': this.windowMs,
      'preRollMs': this.preRollMs,
      'vadThreshold': this.vadThreshold,
    };
  }

  factory FfiHotwordConfig.fromJson(Map<String, dynamic> json) {
    return FfiHotwordConfig(
      threshold: (json['threshold'] as num).toDouble(),
      cooldownMs: (json['cooldownMs'] as num).toInt(),
      stepMs: (json['stepMs'] as num).toInt(),
      windowMs: (json['windowMs'] as num).toInt(),
      preRollMs: (json['preRollMs'] as num).toInt(),
      vadThreshold: (json['vadThreshold'] as num).toDouble(),
    );
  }

  FfiHotwordConfig copyWith({
    double? threshold,
    int? cooldownMs,
    int? stepMs,
    int? windowMs,
    int? preRollMs,
    double? vadThreshold,
  }) {
    return FfiHotwordConfig(
      threshold: threshold ?? this.threshold,
      cooldownMs: cooldownMs ?? this.cooldownMs,
      stepMs: stepMs ?? this.stepMs,
      windowMs: windowMs ?? this.windowMs,
      preRollMs: preRollMs ?? this.preRollMs,
      vadThreshold: vadThreshold ?? this.vadThreshold,
    );
  }

  @override
  String toString() {
    return 'FfiHotwordConfig(threshold: $threshold, cooldownMs: $cooldownMs, stepMs: $stepMs, windowMs: $windowMs, preRollMs: $preRollMs, vadThreshold: $vadThreshold)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiHotwordConfig && threshold == other.threshold && cooldownMs == other.cooldownMs && stepMs == other.stepMs && windowMs == other.windowMs && preRollMs == other.preRollMs && vadThreshold == other.vadThreshold;

  @override
  int get hashCode => Object.hash(threshold, cooldownMs, stepMs, windowMs, preRollMs, vadThreshold);
}

/// Event emitted when a keyword spotting threshold is crossed.
class FfiHotwordEvent {
  const FfiHotwordEvent({
    /// The matched keyword string.
    required this.keyword,
    /// Exact audio stream sample index where the keyword completed.
    required this.sampleOffset,
    /// Audio stream sample index including pre-roll safety margin for downstream ASR.
    required this.commandStartSample,
    /// Timestamp in milliseconds from stream origin where keyword completed.
    required this.timestampMs,
    /// Model confidence probability (0.0 to 1.0).
    required this.confidence,
  });

  /// The matched keyword string.
  final String keyword;
  /// Exact audio stream sample index where the keyword completed.
  final int sampleOffset;
  /// Audio stream sample index including pre-roll safety margin for downstream ASR.
  final int commandStartSample;
  /// Timestamp in milliseconds from stream origin where keyword completed.
  final double timestampMs;
  /// Model confidence probability (0.0 to 1.0).
  final double confidence;

  Map<String, dynamic> toJson() {
    return {
      'keyword': this.keyword,
      'sampleOffset': this.sampleOffset,
      'commandStartSample': this.commandStartSample,
      'timestampMs': this.timestampMs,
      'confidence': this.confidence,
    };
  }

  factory FfiHotwordEvent.fromJson(Map<String, dynamic> json) {
    return FfiHotwordEvent(
      keyword: json['keyword'] as String,
      sampleOffset: (json['sampleOffset'] as num).toInt(),
      commandStartSample: (json['commandStartSample'] as num).toInt(),
      timestampMs: (json['timestampMs'] as num).toDouble(),
      confidence: (json['confidence'] as num).toDouble(),
    );
  }

  FfiHotwordEvent copyWith({
    String? keyword,
    int? sampleOffset,
    int? commandStartSample,
    double? timestampMs,
    double? confidence,
  }) {
    return FfiHotwordEvent(
      keyword: keyword ?? this.keyword,
      sampleOffset: sampleOffset ?? this.sampleOffset,
      commandStartSample: commandStartSample ?? this.commandStartSample,
      timestampMs: timestampMs ?? this.timestampMs,
      confidence: confidence ?? this.confidence,
    );
  }

  @override
  String toString() {
    return 'FfiHotwordEvent(keyword: $keyword, sampleOffset: $sampleOffset, commandStartSample: $commandStartSample, timestampMs: $timestampMs, confidence: $confidence)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiHotwordEvent && keyword == other.keyword && sampleOffset == other.sampleOffset && commandStartSample == other.commandStartSample && timestampMs == other.timestampMs && confidence == other.confidence;

  @override
  int get hashCode => Object.hash(keyword, sampleOffset, commandStartSample, timestampMs, confidence);
}

/// Confidence score for a specific keyword candidate.
class FfiHotwordScore {
  const FfiHotwordScore({
    /// Target keyword string.
    required this.keyword,
    /// Model activation probability between 0.0 and 1.0.
    required this.score,
  });

  /// Target keyword string.
  final String keyword;
  /// Model activation probability between 0.0 and 1.0.
  final double score;

  Map<String, dynamic> toJson() {
    return {
      'keyword': this.keyword,
      'score': this.score,
    };
  }

  factory FfiHotwordScore.fromJson(Map<String, dynamic> json) {
    return FfiHotwordScore(
      keyword: json['keyword'] as String,
      score: (json['score'] as num).toDouble(),
    );
  }

  FfiHotwordScore copyWith({
    String? keyword,
    double? score,
  }) {
    return FfiHotwordScore(
      keyword: keyword ?? this.keyword,
      score: score ?? this.score,
    );
  }

  @override
  String toString() {
    return 'FfiHotwordScore(keyword: $keyword, score: $score)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiHotwordScore && keyword == other.keyword && score == other.score;

  @override
  int get hashCode => Object.hash(keyword, score);
}

/// A detected speech segment with sample and millisecond boundaries.
class FfiSpeechTimestamp {
  const FfiSpeechTimestamp({
    required this.startSample,
    required this.endSample,
    required this.startMs,
    required this.endMs,
  });

  final int startSample;
  final int endSample;
  final double startMs;
  final double endMs;

  Map<String, dynamic> toJson() {
    return {
      'startSample': this.startSample,
      'endSample': this.endSample,
      'startMs': this.startMs,
      'endMs': this.endMs,
    };
  }

  factory FfiSpeechTimestamp.fromJson(Map<String, dynamic> json) {
    return FfiSpeechTimestamp(
      startSample: (json['startSample'] as num).toInt(),
      endSample: (json['endSample'] as num).toInt(),
      startMs: (json['startMs'] as num).toDouble(),
      endMs: (json['endMs'] as num).toDouble(),
    );
  }

  FfiSpeechTimestamp copyWith({
    int? startSample,
    int? endSample,
    double? startMs,
    double? endMs,
  }) {
    return FfiSpeechTimestamp(
      startSample: startSample ?? this.startSample,
      endSample: endSample ?? this.endSample,
      startMs: startMs ?? this.startMs,
      endMs: endMs ?? this.endMs,
    );
  }

  @override
  String toString() {
    return 'FfiSpeechTimestamp(startSample: $startSample, endSample: $endSample, startMs: $startMs, endMs: $endMs)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiSpeechTimestamp && startSample == other.startSample && endSample == other.endSample && startMs == other.startMs && endMs == other.endMs;

  @override
  int get hashCode => Object.hash(startSample, endSample, startMs, endMs);
}

/// Configuration options for batch speech detection and segmentation.
class FfiVadConfig {
  const FfiVadConfig({
    this.threshold = 0.5,
    this.negThreshold = 0.35,
    this.minSpeechDurationMs = 64,
    this.minSilenceDurationMs = 100,
    this.speechPadMs = 30,
  });

  final double threshold;
  final double negThreshold;
  final int minSpeechDurationMs;
  final int minSilenceDurationMs;
  final int speechPadMs;

  Map<String, dynamic> toJson() {
    return {
      'threshold': this.threshold,
      'negThreshold': this.negThreshold,
      'minSpeechDurationMs': this.minSpeechDurationMs,
      'minSilenceDurationMs': this.minSilenceDurationMs,
      'speechPadMs': this.speechPadMs,
    };
  }

  factory FfiVadConfig.fromJson(Map<String, dynamic> json) {
    return FfiVadConfig(
      threshold: json.containsKey('threshold') ? (json['threshold'] as num).toDouble() : 0.5,
      negThreshold: json.containsKey('negThreshold') ? (json['negThreshold'] as num).toDouble() : 0.35,
      minSpeechDurationMs: json.containsKey('minSpeechDurationMs') ? (json['minSpeechDurationMs'] as num).toInt() : 64,
      minSilenceDurationMs: json.containsKey('minSilenceDurationMs') ? (json['minSilenceDurationMs'] as num).toInt() : 100,
      speechPadMs: json.containsKey('speechPadMs') ? (json['speechPadMs'] as num).toInt() : 30,
    );
  }

  FfiVadConfig copyWith({
    double? threshold,
    double? negThreshold,
    int? minSpeechDurationMs,
    int? minSilenceDurationMs,
    int? speechPadMs,
  }) {
    return FfiVadConfig(
      threshold: threshold ?? this.threshold,
      negThreshold: negThreshold ?? this.negThreshold,
      minSpeechDurationMs: minSpeechDurationMs ?? this.minSpeechDurationMs,
      minSilenceDurationMs: minSilenceDurationMs ?? this.minSilenceDurationMs,
      speechPadMs: speechPadMs ?? this.speechPadMs,
    );
  }

  @override
  String toString() {
    return 'FfiVadConfig(threshold: $threshold, negThreshold: $negThreshold, minSpeechDurationMs: $minSpeechDurationMs, minSilenceDurationMs: $minSilenceDurationMs, speechPadMs: $speechPadMs)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiVadConfig && threshold == other.threshold && negThreshold == other.negThreshold && minSpeechDurationMs == other.minSpeechDurationMs && minSilenceDurationMs == other.minSilenceDurationMs && speechPadMs == other.speechPadMs;

  @override
  int get hashCode => Object.hash(threshold, negThreshold, minSpeechDurationMs, minSilenceDurationMs, speechPadMs);
}

/// Per-call decode options. Mirrors [`cera::GenerateOpts`].
///
/// `flush_every_tokens` / `flush_every_ms` are accepted but have no
/// effect under the synchronous [`Session::generate`]; they are
/// meaningful once streaming (foreign-trait `ModalitySink`) lands
/// in a follow-up PR. Including them in the record now keeps the FFI
/// surface stable across that transition.
class GenerateOpts {
  const GenerateOpts({
    this.maxTokens = 256,
    this.temperature = 0.7,
    this.topP = 0.9,
    this.topK = 40,
    /// Min-p (relative) nucleus cutoff: drop tokens below `min_p * p_max`. `0.0`
    /// disables it. Honored in the stochastic path.
    this.minP = 0.05,
    /// Repetition penalty over tokens generated this call. `1.0` disables it.
    /// Honored in the stochastic path (greedy/argmax decoding is unaffected).
    this.repetitionPenalty = 1.1,
    /// Early-stop IDs (EOS / instruction markers / end-of-turn).
    this.stopTokens = const [],
    /// Ignore end-of-generation: EOS and `stop_tokens` are not honored, so
    /// decode always runs to `max_tokens`. For benchmark loops that must
    /// cover an exact token count.
    this.ignoreEos = false,
    /// Optional GBNF grammar **source text** constraining the output (e.g. a
    /// JSON grammar). When absent (the default), decoding is unconstrained. The
    /// grammar is compiled on the Rust side when generation starts; a malformed
    /// grammar is reported as a `GrammarParse` error.
    this.grammar = null,
    /// Lazy-grammar trigger token ids (tool calling). When non-empty and
    /// `grammar` is set, the grammar stays inactive until the model emits one
    /// of these tokens (e.g. the tool-call start marker from
    /// [`CeraEngine::tool_call_start_token`]), then constrains the call and
    /// deactivates on completion. Empty -> `grammar` is active from the start.
    this.grammarTriggerTokens = const [],
    /// Ignored under synchronous generate; reserved for streaming.
    this.flushEveryTokens = 16,
    /// Ignored under synchronous generate; reserved for streaming.
    this.flushEveryMs = 50,
    /// Optional speculative decoding configuration (prompt-lookup drafting).
    /// When set, runs prompt-lookup speculative drafting to accelerate greedy decoding.
    this.spec = null,
  });

  final int maxTokens;
  final double temperature;
  final double topP;
  final int topK;
  /// Min-p (relative) nucleus cutoff: drop tokens below `min_p * p_max`. `0.0`
  /// disables it. Honored in the stochastic path.
  final double minP;
  /// Repetition penalty over tokens generated this call. `1.0` disables it.
  /// Honored in the stochastic path (greedy/argmax decoding is unaffected).
  final double repetitionPenalty;
  /// Early-stop IDs (EOS / instruction markers / end-of-turn).
  final List<int> stopTokens;
  /// Ignore end-of-generation: EOS and `stop_tokens` are not honored, so
  /// decode always runs to `max_tokens`. For benchmark loops that must
  /// cover an exact token count.
  final bool ignoreEos;
  /// Optional GBNF grammar **source text** constraining the output (e.g. a
  /// JSON grammar). When absent (the default), decoding is unconstrained. The
  /// grammar is compiled on the Rust side when generation starts; a malformed
  /// grammar is reported as a `GrammarParse` error.
  final String? grammar;
  /// Lazy-grammar trigger token ids (tool calling). When non-empty and
  /// `grammar` is set, the grammar stays inactive until the model emits one
  /// of these tokens (e.g. the tool-call start marker from
  /// [`CeraEngine::tool_call_start_token`]), then constrains the call and
  /// deactivates on completion. Empty -> `grammar` is active from the start.
  final List<int> grammarTriggerTokens;
  /// Ignored under synchronous generate; reserved for streaming.
  final int flushEveryTokens;
  /// Ignored under synchronous generate; reserved for streaming.
  final int flushEveryMs;
  /// Optional speculative decoding configuration (prompt-lookup drafting).
  /// When set, runs prompt-lookup speculative drafting to accelerate greedy decoding.
  final SpecDecodeConfig? spec;

  Map<String, dynamic> toJson() {
    return {
      'maxTokens': this.maxTokens,
      'temperature': this.temperature,
      'topP': this.topP,
      'topK': this.topK,
      'minP': this.minP,
      'repetitionPenalty': this.repetitionPenalty,
      'stopTokens': this.stopTokens,
      'ignoreEos': this.ignoreEos,
      'grammar': this.grammar,
      'grammarTriggerTokens': this.grammarTriggerTokens,
      'flushEveryTokens': this.flushEveryTokens,
      'flushEveryMs': this.flushEveryMs,
      'spec': this.spec == null ? null : (() { final __tmp = this.spec!; return __tmp.toJson(); })(),
    };
  }

  factory GenerateOpts.fromJson(Map<String, dynamic> json) {
    return GenerateOpts(
      maxTokens: json.containsKey('maxTokens') ? (json['maxTokens'] as num).toInt() : 256,
      temperature: json.containsKey('temperature') ? (json['temperature'] as num).toDouble() : 0.7,
      topP: json.containsKey('topP') ? (json['topP'] as num).toDouble() : 0.9,
      topK: json.containsKey('topK') ? (json['topK'] as num).toInt() : 40,
      minP: json.containsKey('minP') ? (json['minP'] as num).toDouble() : 0.05,
      repetitionPenalty: json.containsKey('repetitionPenalty') ? (json['repetitionPenalty'] as num).toDouble() : 1.1,
      stopTokens: json.containsKey('stopTokens') ? (json['stopTokens'] as List).map((item) => (item as num).toInt()).toList() : const [],
      ignoreEos: json.containsKey('ignoreEos') ? json['ignoreEos'] as bool : false,
      grammar: json.containsKey('grammar') ? json['grammar'] == null ? null : json['grammar'] as String : null,
      grammarTriggerTokens: json.containsKey('grammarTriggerTokens') ? (json['grammarTriggerTokens'] as List).map((item) => (item as num).toInt()).toList() : const [],
      flushEveryTokens: json.containsKey('flushEveryTokens') ? (json['flushEveryTokens'] as num).toInt() : 16,
      flushEveryMs: json.containsKey('flushEveryMs') ? (json['flushEveryMs'] as num).toInt() : 50,
      spec: json.containsKey('spec') ? json['spec'] == null ? null : (() { final __tmp = json['spec']; return SpecDecodeConfig.fromJson(__tmp as Map<String, dynamic>); })() : null,
    );
  }

  GenerateOpts copyWith({
    int? maxTokens,
    double? temperature,
    double? topP,
    int? topK,
    double? minP,
    double? repetitionPenalty,
    List<int>? stopTokens,
    bool? ignoreEos,
    Object? grammar = _sentinel,
    List<int>? grammarTriggerTokens,
    int? flushEveryTokens,
    int? flushEveryMs,
    Object? spec = _sentinel,
  }) {
    return GenerateOpts(
      maxTokens: maxTokens ?? this.maxTokens,
      temperature: temperature ?? this.temperature,
      topP: topP ?? this.topP,
      topK: topK ?? this.topK,
      minP: minP ?? this.minP,
      repetitionPenalty: repetitionPenalty ?? this.repetitionPenalty,
      stopTokens: stopTokens ?? this.stopTokens,
      ignoreEos: ignoreEos ?? this.ignoreEos,
      grammar: grammar == _sentinel ? this.grammar : grammar as String?,
      grammarTriggerTokens: grammarTriggerTokens ?? this.grammarTriggerTokens,
      flushEveryTokens: flushEveryTokens ?? this.flushEveryTokens,
      flushEveryMs: flushEveryMs ?? this.flushEveryMs,
      spec: spec == _sentinel ? this.spec : spec as SpecDecodeConfig?,
    );
  }

  @override
  String toString() {
    return 'GenerateOpts(maxTokens: $maxTokens, temperature: $temperature, topP: $topP, topK: $topK, minP: $minP, repetitionPenalty: $repetitionPenalty, stopTokens: $stopTokens, ignoreEos: $ignoreEos, grammar: $grammar, grammarTriggerTokens: $grammarTriggerTokens, flushEveryTokens: $flushEveryTokens, flushEveryMs: $flushEveryMs, spec: $spec)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is GenerateOpts && maxTokens == other.maxTokens && temperature == other.temperature && topP == other.topP && topK == other.topK && minP == other.minP && repetitionPenalty == other.repetitionPenalty && stopTokens == other.stopTokens && ignoreEos == other.ignoreEos && grammar == other.grammar && grammarTriggerTokens == other.grammarTriggerTokens && flushEveryTokens == other.flushEveryTokens && flushEveryMs == other.flushEveryMs && spec == other.spec;

  @override
  int get hashCode => Object.hash(maxTokens, temperature, topP, topK, minP, repetitionPenalty, stopTokens, ignoreEos, grammar, grammarTriggerTokens, flushEveryTokens, flushEveryMs, spec);
}

/// Bundle of everything a synchronous `generate` call produces:
/// the generated text string, token IDs, and decode summary.
class GenerateOutput {
  const GenerateOutput({
    /// Generated text (UTF-8 decoded).
    required this.text,
    /// Generated token IDs, in order, not including any prompt tokens.
    required this.tokens,
    required this.summary,
  });

  /// Generated text (UTF-8 decoded).
  final String text;
  /// Generated token IDs, in order, not including any prompt tokens.
  final List<int> tokens;
  final GenerateSummary summary;

  Map<String, dynamic> toJson() {
    return {
      'text': this.text,
      'tokens': this.tokens,
      'summary': this.summary.toJson(),
    };
  }

  factory GenerateOutput.fromJson(Map<String, dynamic> json) {
    return GenerateOutput(
      text: json['text'] as String,
      tokens: (json['tokens'] as List).map((item) => (item as num).toInt()).toList(),
      summary: GenerateSummary.fromJson(json['summary'] as Map<String, dynamic>),
    );
  }

  GenerateOutput copyWith({
    String? text,
    List<int>? tokens,
    GenerateSummary? summary,
  }) {
    return GenerateOutput(
      text: text ?? this.text,
      tokens: tokens ?? this.tokens,
      summary: summary ?? this.summary,
    );
  }

  @override
  String toString() {
    return 'GenerateOutput(text: $text, tokens: $tokens, summary: $summary)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is GenerateOutput && text == other.text && tokens == other.tokens && summary == other.summary;

  @override
  int get hashCode => Object.hash(text, tokens, summary);
}

/// Decode-run metadata. Mirrors [`cera::GenerateSummary`].
class GenerateSummary {
  const GenerateSummary({
    required this.tokensGenerated,
    required this.promptEvalTokens,
    required this.promptEvalMs,
    required this.decodeMs,
    required this.totalDurationMs,
    required this.decodeTokPerSec,
    required this.promptEvalTokPerSec,
    required this.finishReason,
  });

  final int tokensGenerated;
  final int promptEvalTokens;
  final int promptEvalMs;
  final int decodeMs;
  final int totalDurationMs;
  final double decodeTokPerSec;
  final double promptEvalTokPerSec;
  final FinishReason finishReason;

  Map<String, dynamic> toJson() {
    return {
      'tokensGenerated': this.tokensGenerated,
      'promptEvalTokens': this.promptEvalTokens,
      'promptEvalMs': this.promptEvalMs,
      'decodeMs': this.decodeMs,
      'totalDurationMs': this.totalDurationMs,
      'decodeTokPerSec': this.decodeTokPerSec,
      'promptEvalTokPerSec': this.promptEvalTokPerSec,
      'finishReason': FinishReasonFfiCodec.encode(this.finishReason),
    };
  }

  factory GenerateSummary.fromJson(Map<String, dynamic> json) {
    return GenerateSummary(
      tokensGenerated: (json['tokensGenerated'] as num).toInt(),
      promptEvalTokens: (json['promptEvalTokens'] as num).toInt(),
      promptEvalMs: (json['promptEvalMs'] as num).toInt(),
      decodeMs: (json['decodeMs'] as num).toInt(),
      totalDurationMs: (json['totalDurationMs'] as num).toInt(),
      decodeTokPerSec: (json['decodeTokPerSec'] as num).toDouble(),
      promptEvalTokPerSec: (json['promptEvalTokPerSec'] as num).toDouble(),
      finishReason: FinishReasonFfiCodec.decode(json['finishReason'] as String),
    );
  }

  GenerateSummary copyWith({
    int? tokensGenerated,
    int? promptEvalTokens,
    int? promptEvalMs,
    int? decodeMs,
    int? totalDurationMs,
    double? decodeTokPerSec,
    double? promptEvalTokPerSec,
    FinishReason? finishReason,
  }) {
    return GenerateSummary(
      tokensGenerated: tokensGenerated ?? this.tokensGenerated,
      promptEvalTokens: promptEvalTokens ?? this.promptEvalTokens,
      promptEvalMs: promptEvalMs ?? this.promptEvalMs,
      decodeMs: decodeMs ?? this.decodeMs,
      totalDurationMs: totalDurationMs ?? this.totalDurationMs,
      decodeTokPerSec: decodeTokPerSec ?? this.decodeTokPerSec,
      promptEvalTokPerSec: promptEvalTokPerSec ?? this.promptEvalTokPerSec,
      finishReason: finishReason ?? this.finishReason,
    );
  }

  @override
  String toString() {
    return 'GenerateSummary(tokensGenerated: $tokensGenerated, promptEvalTokens: $promptEvalTokens, promptEvalMs: $promptEvalMs, decodeMs: $decodeMs, totalDurationMs: $totalDurationMs, decodeTokPerSec: $decodeTokPerSec, promptEvalTokPerSec: $promptEvalTokPerSec, finishReason: $finishReason)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is GenerateSummary && tokensGenerated == other.tokensGenerated && promptEvalTokens == other.promptEvalTokens && promptEvalMs == other.promptEvalMs && decodeMs == other.decodeMs && totalDurationMs == other.totalDurationMs && decodeTokPerSec == other.decodeTokPerSec && promptEvalTokPerSec == other.promptEvalTokPerSec && finishReason == other.finishReason;

  @override
  int get hashCode => Object.hash(tokensGenerated, promptEvalTokens, promptEvalMs, decodeMs, totalDurationMs, decodeTokPerSec, promptEvalTokPerSec, finishReason);
}

/// One bundle published on `huggingface.co/LiquidAI/LeapBundles`: the
/// model directory plus every per-quant manifest inside it. Feed
/// `name` and one element of `quants` straight to
/// [`CeraEngine::from_bundle_id`].
///
/// Both fields are sorted ascending, so a menu built from this list is
/// stable across runs even if the upstream API reorders its response.
class LeapBundleEntry {
  const LeapBundleEntry({
    required this.name,
    required this.quants,
  });

  final String name;
  final List<String> quants;

  Map<String, dynamic> toJson() {
    return {
      'name': this.name,
      'quants': this.quants,
    };
  }

  factory LeapBundleEntry.fromJson(Map<String, dynamic> json) {
    return LeapBundleEntry(
      name: json['name'] as String,
      quants: (json['quants'] as List).map((item) => item as String).toList(),
    );
  }

  LeapBundleEntry copyWith({
    String? name,
    List<String>? quants,
  }) {
    return LeapBundleEntry(
      name: name ?? this.name,
      quants: quants ?? this.quants,
    );
  }

  @override
  String toString() {
    return 'LeapBundleEntry(name: $name, quants: $quants)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is LeapBundleEntry && name == other.name && quants == other.quants;

  @override
  int get hashCode => Object.hash(name, quants);
}

/// Modality support flags for a loaded model. Mirrors
/// [`cera::ModalityCapabilities`].
class ModalityCapabilities {
  const ModalityCapabilities({
    required this.textIn,
    required this.textOut,
    required this.imageIn,
    required this.audioIn,
    required this.audioOut,
  });

  final bool textIn;
  final bool textOut;
  final bool imageIn;
  final bool audioIn;
  final bool audioOut;

  Map<String, dynamic> toJson() {
    return {
      'textIn': this.textIn,
      'textOut': this.textOut,
      'imageIn': this.imageIn,
      'audioIn': this.audioIn,
      'audioOut': this.audioOut,
    };
  }

  factory ModalityCapabilities.fromJson(Map<String, dynamic> json) {
    return ModalityCapabilities(
      textIn: json['textIn'] as bool,
      textOut: json['textOut'] as bool,
      imageIn: json['imageIn'] as bool,
      audioIn: json['audioIn'] as bool,
      audioOut: json['audioOut'] as bool,
    );
  }

  ModalityCapabilities copyWith({
    bool? textIn,
    bool? textOut,
    bool? imageIn,
    bool? audioIn,
    bool? audioOut,
  }) {
    return ModalityCapabilities(
      textIn: textIn ?? this.textIn,
      textOut: textOut ?? this.textOut,
      imageIn: imageIn ?? this.imageIn,
      audioIn: audioIn ?? this.audioIn,
      audioOut: audioOut ?? this.audioOut,
    );
  }

  @override
  String toString() {
    return 'ModalityCapabilities(textIn: $textIn, textOut: $textOut, imageIn: $imageIn, audioIn: $audioIn, audioOut: $audioOut)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ModalityCapabilities && textIn == other.textIn && textOut == other.textOut && imageIn == other.imageIn && audioIn == other.audioIn && audioOut == other.audioOut;

  @override
  int get hashCode => Object.hash(textIn, textOut, imageIn, audioIn, audioOut);
}

/// Short summary of a loaded model. Mirrors [`cera::ModelMetadata`].
class ModelMetadata {
  const ModelMetadata({
    required this.architecture,
    required this.maxSeqLen,
    required this.vocabSize,
    required this.hasChatTemplate,
    required this.quantization,
    /// Mirror of GGUF `tokenizer.ggml.add_bos_token`. Consumers that
    /// want to insert a BOS at the head of a raw prompt should honor it —
    /// or, better, tokenize via `encode_text_special`, which applies both
    /// this and `add_eos_token`.
    required this.addBosToken,
    /// Mirror of GGUF `tokenizer.ggml.add_eos_token`. See `add_bos_token`.
    required this.addEosToken,
    /// SIMD backend tier the runtime resolved for this host (e.g.
    /// `"neon+dotprod"`, `"avx2"`, `"scalar"`). A host property, not
    /// model-specific — surfaced here so consumers fetching metadata also
    /// get backend diagnostics for telemetry / bug reports. For the full
    /// feature list, see [`cpu_backend_report`].
    required this.cpuBackend,
  });

  final String architecture;
  final int maxSeqLen;
  final int vocabSize;
  final bool hasChatTemplate;
  final String quantization;
  /// Mirror of GGUF `tokenizer.ggml.add_bos_token`. Consumers that
  /// want to insert a BOS at the head of a raw prompt should honor it —
  /// or, better, tokenize via `encode_text_special`, which applies both
  /// this and `add_eos_token`.
  final bool addBosToken;
  /// Mirror of GGUF `tokenizer.ggml.add_eos_token`. See `add_bos_token`.
  final bool addEosToken;
  /// SIMD backend tier the runtime resolved for this host (e.g.
  /// `"neon+dotprod"`, `"avx2"`, `"scalar"`). A host property, not
  /// model-specific — surfaced here so consumers fetching metadata also
  /// get backend diagnostics for telemetry / bug reports. For the full
  /// feature list, see [`cpu_backend_report`].
  final String cpuBackend;

  Map<String, dynamic> toJson() {
    return {
      'architecture': this.architecture,
      'maxSeqLen': this.maxSeqLen,
      'vocabSize': this.vocabSize,
      'hasChatTemplate': this.hasChatTemplate,
      'quantization': this.quantization,
      'addBosToken': this.addBosToken,
      'addEosToken': this.addEosToken,
      'cpuBackend': this.cpuBackend,
    };
  }

  factory ModelMetadata.fromJson(Map<String, dynamic> json) {
    return ModelMetadata(
      architecture: json['architecture'] as String,
      maxSeqLen: (json['maxSeqLen'] as num).toInt(),
      vocabSize: (json['vocabSize'] as num).toInt(),
      hasChatTemplate: json['hasChatTemplate'] as bool,
      quantization: json['quantization'] as String,
      addBosToken: json['addBosToken'] as bool,
      addEosToken: json['addEosToken'] as bool,
      cpuBackend: json['cpuBackend'] as String,
    );
  }

  ModelMetadata copyWith({
    String? architecture,
    int? maxSeqLen,
    int? vocabSize,
    bool? hasChatTemplate,
    String? quantization,
    bool? addBosToken,
    bool? addEosToken,
    String? cpuBackend,
  }) {
    return ModelMetadata(
      architecture: architecture ?? this.architecture,
      maxSeqLen: maxSeqLen ?? this.maxSeqLen,
      vocabSize: vocabSize ?? this.vocabSize,
      hasChatTemplate: hasChatTemplate ?? this.hasChatTemplate,
      quantization: quantization ?? this.quantization,
      addBosToken: addBosToken ?? this.addBosToken,
      addEosToken: addEosToken ?? this.addEosToken,
      cpuBackend: cpuBackend ?? this.cpuBackend,
    );
  }

  @override
  String toString() {
    return 'ModelMetadata(architecture: $architecture, maxSeqLen: $maxSeqLen, vocabSize: $vocabSize, hasChatTemplate: $hasChatTemplate, quantization: $quantization, addBosToken: $addBosToken, addEosToken: $addEosToken, cpuBackend: $cpuBackend)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ModelMetadata && architecture == other.architecture && maxSeqLen == other.maxSeqLen && vocabSize == other.vocabSize && hasChatTemplate == other.hasChatTemplate && quantization == other.quantization && addBosToken == other.addBosToken && addEosToken == other.addEosToken && cpuBackend == other.cpuBackend;

  @override
  int get hashCode => Object.hash(architecture, maxSeqLen, vocabSize, hasChatTemplate, quantization, addBosToken, addEosToken, cpuBackend);
}

/// Per-session configuration. Mirrors [`cera::SessionConfig`].
class SessionConfig {
  const SessionConfig({
    /// Cap on total tokens held in KV. `None` → model's default
    /// `max_seq_len`.
    this.maxSeqLen = null,
    /// KV cache compression mode. `None` → no compression (the default).
    this.kvCompression = null,
    /// Pinned-prefix length for Phase-1.5 context shift on overflow.
    /// `0` disables shift; overflow returns `ContextOverflow` error.
    this.nKeep = 0,
    /// Deterministic sampling seed. `None` = fresh entropy per call.
    this.seed = null,
    /// Chunked-prefill ubatch size. `0` = monolithic prefill.
    this.ubatchSize = 512,
    /// Whether to prefer GPU depthformer for audio decoder generation.
    this.gpuDepthformer = false,
  });

  /// Cap on total tokens held in KV. `None` → model's default
  /// `max_seq_len`.
  final int? maxSeqLen;
  /// KV cache compression mode. `None` → no compression (the default).
  final KvCompression? kvCompression;
  /// Pinned-prefix length for Phase-1.5 context shift on overflow.
  /// `0` disables shift; overflow returns `ContextOverflow` error.
  final int nKeep;
  /// Deterministic sampling seed. `None` = fresh entropy per call.
  final int? seed;
  /// Chunked-prefill ubatch size. `0` = monolithic prefill.
  final int ubatchSize;
  /// Whether to prefer GPU depthformer for audio decoder generation.
  final bool gpuDepthformer;

  Map<String, dynamic> toJson() {
    return {
      'maxSeqLen': this.maxSeqLen,
      'kvCompression': this.kvCompression == null ? null : (() { final __tmp = this.kvCompression!; return KvCompressionFfiCodec.encode(__tmp); })(),
      'nKeep': this.nKeep,
      'seed': this.seed,
      'ubatchSize': this.ubatchSize,
      'gpuDepthformer': this.gpuDepthformer,
    };
  }

  factory SessionConfig.fromJson(Map<String, dynamic> json) {
    return SessionConfig(
      maxSeqLen: json.containsKey('maxSeqLen') ? json['maxSeqLen'] == null ? null : (json['maxSeqLen'] as num).toInt() : null,
      kvCompression: json.containsKey('kvCompression') ? json['kvCompression'] == null ? null : (() { final __tmp = json['kvCompression']; return KvCompressionFfiCodec.decode(__tmp as String); })() : null,
      nKeep: json.containsKey('nKeep') ? (json['nKeep'] as num).toInt() : 0,
      seed: json.containsKey('seed') ? json['seed'] == null ? null : (json['seed'] as num).toInt() : null,
      ubatchSize: json.containsKey('ubatchSize') ? (json['ubatchSize'] as num).toInt() : 512,
      gpuDepthformer: json.containsKey('gpuDepthformer') ? json['gpuDepthformer'] as bool : false,
    );
  }

  SessionConfig copyWith({
    Object? maxSeqLen = _sentinel,
    Object? kvCompression = _sentinel,
    int? nKeep,
    Object? seed = _sentinel,
    int? ubatchSize,
    bool? gpuDepthformer,
  }) {
    return SessionConfig(
      maxSeqLen: maxSeqLen == _sentinel ? this.maxSeqLen : maxSeqLen as int?,
      kvCompression: kvCompression == _sentinel ? this.kvCompression : kvCompression as KvCompression?,
      nKeep: nKeep ?? this.nKeep,
      seed: seed == _sentinel ? this.seed : seed as int?,
      ubatchSize: ubatchSize ?? this.ubatchSize,
      gpuDepthformer: gpuDepthformer ?? this.gpuDepthformer,
    );
  }

  @override
  String toString() {
    return 'SessionConfig(maxSeqLen: $maxSeqLen, kvCompression: $kvCompression, nKeep: $nKeep, seed: $seed, ubatchSize: $ubatchSize, gpuDepthformer: $gpuDepthformer)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SessionConfig && maxSeqLen == other.maxSeqLen && kvCompression == other.kvCompression && nKeep == other.nKeep && seed == other.seed && ubatchSize == other.ubatchSize && gpuDepthformer == other.gpuDepthformer;

  @override
  int get hashCode => Object.hash(maxSeqLen, kvCompression, nKeep, seed, ubatchSize, gpuDepthformer);
}

/// Speculative decoding configuration for prompt-lookup drafting. Mirrors [`cera::SpecDecode`].
///
/// Prompt-lookup drafting matches trailing n-grams in history to propose draft candidates
/// and verifies them in a single batched prefill step. This is optimal for memory-bandwidth-bound
/// CPU inference; on high-throughput GPU backends, verification overhead may reduce net speedup.
class SpecDecodeConfig {
  const SpecDecodeConfig({
    /// Length of the trailing n-gram matched to locate a draft. Defaults to 2.
    this.ngram = 2,
    /// Maximum draft length verified per round (speculation depth). Defaults to 6.
    this.k = 6,
  });

  /// Length of the trailing n-gram matched to locate a draft. Defaults to 2.
  final int ngram;
  /// Maximum draft length verified per round (speculation depth). Defaults to 6.
  final int k;

  Map<String, dynamic> toJson() {
    return {
      'ngram': this.ngram,
      'k': this.k,
    };
  }

  factory SpecDecodeConfig.fromJson(Map<String, dynamic> json) {
    return SpecDecodeConfig(
      ngram: json.containsKey('ngram') ? (json['ngram'] as num).toInt() : 2,
      k: json.containsKey('k') ? (json['k'] as num).toInt() : 6,
    );
  }

  SpecDecodeConfig copyWith({
    int? ngram,
    int? k,
  }) {
    return SpecDecodeConfig(
      ngram: ngram ?? this.ngram,
      k: k ?? this.k,
    );
  }

  @override
  String toString() {
    return 'SpecDecodeConfig(ngram: $ngram, k: $k)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SpecDecodeConfig && ngram == other.ngram && k == other.k;

  @override
  int get hashCode => Object.hash(ngram, k);
}

/// A tool call parsed from model output. Mirrors [`cera::tools::ToolCall`];
/// `arguments_json` is the call's arguments encoded as a JSON string.
class ToolCall {
  const ToolCall({
    required this.name,
    /// The call's arguments as a JSON string — normally an object
    /// (e.g. `{"city":"Paris"}`), but a malformed Hermes/Qwen reply may pass
    /// through a non-object value, so decode defensively.
    required this.argumentsJson,
  });

  final String name;
  /// The call's arguments as a JSON string — normally an object
  /// (e.g. `{"city":"Paris"}`), but a malformed Hermes/Qwen reply may pass
  /// through a non-object value, so decode defensively.
  final String argumentsJson;

  Map<String, dynamic> toJson() {
    return {
      'name': this.name,
      'argumentsJson': this.argumentsJson,
    };
  }

  factory ToolCall.fromJson(Map<String, dynamic> json) {
    return ToolCall(
      name: json['name'] as String,
      argumentsJson: json['argumentsJson'] as String,
    );
  }

  ToolCall copyWith({
    String? name,
    String? argumentsJson,
  }) {
    return ToolCall(
      name: name ?? this.name,
      argumentsJson: argumentsJson ?? this.argumentsJson,
    );
  }

  @override
  String toString() {
    return 'ToolCall(name: $name, argumentsJson: $argumentsJson)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ToolCall && name == other.name && argumentsJson == other.argumentsJson;

  @override
  int get hashCode => Object.hash(name, argumentsJson);
}

/// A tool the model may call. Mirrors [`cera::tools::ToolDef`], but the
/// JSON Schema for the arguments crosses the boundary as a JSON **string**
/// (`parameters_json`) since UniFFI has no arbitrary-JSON type. An empty
/// `parameters_json` means "no parameters".
class ToolDef {
  const ToolDef({
    required this.name,
    required this.description,
    /// JSON Schema object for the arguments, as a JSON string (e.g.
    /// `{"type":"object","properties":{…},"required":[…]}`). Empty → none.
    required this.parametersJson,
  });

  final String name;
  final String? description;
  /// JSON Schema object for the arguments, as a JSON string (e.g.
  /// `{"type":"object","properties":{…},"required":[…]}`). Empty → none.
  final String parametersJson;

  Map<String, dynamic> toJson() {
    return {
      'name': this.name,
      'description': this.description,
      'parametersJson': this.parametersJson,
    };
  }

  factory ToolDef.fromJson(Map<String, dynamic> json) {
    return ToolDef(
      name: json['name'] as String,
      description: json['description'] == null ? null : json['description'] as String,
      parametersJson: json['parametersJson'] as String,
    );
  }

  ToolDef copyWith({
    String? name,
    Object? description = _sentinel,
    String? parametersJson,
  }) {
    return ToolDef(
      name: name ?? this.name,
      description: description == _sentinel ? this.description : description as String?,
      parametersJson: parametersJson ?? this.parametersJson,
    );
  }

  @override
  String toString() {
    return 'ToolDef(name: $name, description: $description, parametersJson: $parametersJson)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ToolDef && name == other.name && description == other.description && parametersJson == other.parametersJson;

  @override
  int get hashCode => Object.hash(name, description, parametersJson);
}

/// User-facing multimodal input envelope.
class UserMessage {
  const UserMessage({
    this.text = null,
    this.images = const [],
    this.audio = null,
  });

  final String? text;
  final List<Uint8List> images;
  final AudioInput? audio;

  Map<String, dynamic> toJson() {
    return {
      'text': this.text,
      'images': this.images.map((item) => base64Encode(item)).toList(),
      'audio': this.audio == null ? null : (() { final __tmp = this.audio!; return __tmp.toJson(); })(),
    };
  }

  factory UserMessage.fromJson(Map<String, dynamic> json) {
    return UserMessage(
      text: json.containsKey('text') ? json['text'] == null ? null : json['text'] as String : null,
      images: json.containsKey('images') ? (json['images'] as List).map((item) => base64Decode(item as String)).toList() : const [],
      audio: json.containsKey('audio') ? json['audio'] == null ? null : (() { final __tmp = json['audio']; return AudioInput.fromJson(__tmp as Map<String, dynamic>); })() : null,
    );
  }

  UserMessage copyWith({
    Object? text = _sentinel,
    List<Uint8List>? images,
    Object? audio = _sentinel,
  }) {
    return UserMessage(
      text: text == _sentinel ? this.text : text as String?,
      images: images ?? this.images,
      audio: audio == _sentinel ? this.audio : audio as AudioInput?,
    );
  }

  @override
  String toString() {
    return 'UserMessage(text: $text, images: $images, audio: $audio)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is UserMessage && text == other.text && images == other.images && audio == other.audio;

  @override
  int get hashCode => Object.hash(text, images, audio);
}

/// Compute-backend selector. Mirrors [`cera::BackendPreference`];
/// kept as a separate type so the `cera` crate doesn't carry UniFFI
/// annotations.
enum BackendPreference {
  /// Probe Metal → GPU → CPU at load time.
  auto,
  cpu,
  /// `wgpu` (Vulkan / Metal / DX12). Requires the `gpu` feature.
  gpu,
  /// Native Metal. Requires the `metal` feature + macOS.
  metal,
}

/// Typed error surface for `cera-ffi`. Mirrors [`cera::CeraError`] one-
/// to-one so foreign callers can pattern-match on error class (Kotlin
/// `when`, Swift `switch`, Python `match`) instead of string-sniffing
/// a generic message.
///
/// `Backend` is **not** a silent fallback for unmapped `cera::CeraError`
/// variants — the `From<CeraError>` impl is exhaustive, so adding a
/// new cera variant breaks compilation here. `Backend` exists solely
/// for FFI-internal errors that have no cera analog: `JoinError` from
/// a panicking `spawn_blocking` task, a poisoned `Session::inner`
/// mutex, 32-bit `u64 → usize` overflow in `EngineConfig::try_from`.
///
/// Every variant carries the data needed to act on it:
/// `ContextOverflow` exposes `max_seq_len` and `by` so callers can
/// reset or truncate rather than re-reading the message;
/// `UnsupportedInferenceType` exposes the offending value;
/// `Io` preserves the underlying OS error message as a string since
/// `io::Error` isn't UniFFI-marshallable.
///
/// `#[error(...)]` format strings match `cera::CeraError` exactly for
/// every shared variant, so `Display` output is identical whether the
/// error originates from cera directly or routes through the FFI
/// wrapper. Pinned by `ffi_error_display_matches_cera_error_for_every_shared_variant`.
sealed class FfiError {
  const FfiError();
}

/// The loaded model doesn't support the modality the caller
/// requested (e.g. `append_audio` on a text-only LLM).
final class FfiErrorUnsupportedModality extends FfiError {
  const FfiErrorUnsupportedModality();

  @override
  String toString() {
    return 'FfiErrorUnsupportedModality()';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorUnsupportedModality;

  @override
  int get hashCode => runtimeType.hashCode;
}

/// The manifest's `inference_type` is one cera doesn't recognize
/// at this version. Field carries the offending string.
final class FfiErrorUnsupportedInferenceType extends FfiError {
  const FfiErrorUnsupportedInferenceType({
    required this.inferenceType,
  });
  final String inferenceType;

  @override
  String toString() {
    return 'FfiErrorUnsupportedInferenceType(inferenceType: $inferenceType)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorUnsupportedInferenceType && inferenceType == other.inferenceType;

  @override
  int get hashCode => inferenceType.hashCode;
}

/// A concurrent `generate*` call is already in flight on this
/// session. Rust side guards with a mutex; this surfaces when the
/// FFI detects contention.
final class FfiErrorBusy extends FfiError {
  const FfiErrorBusy();

  @override
  String toString() {
    return 'FfiErrorBusy()';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorBusy;

  @override
  int get hashCode => runtimeType.hashCode;
}

/// The caller (or the cancel-on-drop guard) flipped the cancel
/// atomic mid-call. Surfaces from `append_text`, `append_tokens`,
/// and `append_audio` when chunked prefill detects the cancel
/// flag between micro-batches and aborts (see
/// [`cera::Session::append_tokens`] for the chunked-prefill
/// mechanism). Call [`Session::clear_cancel`] to reset the flag
/// so the next call can proceed.
///
/// `generate` reports cancellation via a different path: the
/// call still returns `Ok` with a [`GenerateOutput`] whose
/// `finish_reason` is set to `Cancelled`. Two paths because
/// chunked prefill has nothing useful to return on cancel (no
/// decoded tokens) while decode has accumulated tokens worth
/// preserving.
final class FfiErrorCancelled extends FfiError {
  const FfiErrorCancelled();

  @override
  String toString() {
    return 'FfiErrorCancelled()';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorCancelled;

  @override
  int get hashCode => runtimeType.hashCode;
}

/// The context window is full and the session can't shift to make
/// room (e.g. `n_keep == 0`, TurboQuant caches, or the active
/// model doesn't support rope-shift). `max_seq_len` is the cap
/// that was hit; `by` is the overshoot in tokens.
final class FfiErrorContextOverflow extends FfiError {
  const FfiErrorContextOverflow({
    required this.maxSeqLen,
    required this.by,
  });
  final int maxSeqLen;
  final int by;

  @override
  String toString() {
    return 'FfiErrorContextOverflow(maxSeqLen: $maxSeqLen, by: $by)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorContextOverflow && maxSeqLen == other.maxSeqLen && by == other.by;

  @override
  int get hashCode => Object.hash(maxSeqLen, by);
}

/// Input buffer was empty (e.g. `append_text("")`, or decode with
/// no prefill state).
final class FfiErrorEmptyInput extends FfiError {
  const FfiErrorEmptyInput();

  @override
  String toString() {
    return 'FfiErrorEmptyInput()';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorEmptyInput;

  @override
  int get hashCode => runtimeType.hashCode;
}

/// Filesystem / mmap / network error surfaced from cera. The
/// underlying `io::Error` isn't marshallable, so the message is
/// flattened to a string. Callers that need the raw kind should
/// parse the `detail` field or open an issue to request a typed
/// field.
///
/// Field is named `detail` rather than `message` because UniFFI's
/// 0.31 Kotlin generator emits `class Io(val `message`) : FfiException()`
/// AND `override val message` in the body when the field is literally
/// named `message`, producing a "conflicting declarations" error
/// (the constructor param collides with the inherited
/// `Throwable.message` override). Renaming to `detail` sidesteps
/// the collision.
///
/// Format string matches `cera::CeraError::Io`'s `"io: {0}"` so
/// foreign `.toString()` / `String(describing:)` gives the same
/// output Rust consumers see.
final class FfiErrorIo extends FfiError {
  const FfiErrorIo({
    required this.detail,
  });
  final String detail;

  @override
  String toString() {
    return 'FfiErrorIo(detail: $detail)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorIo && detail == other.detail;

  @override
  int get hashCode => detail.hashCode;
}

/// FFI-internal error with no cera analog: `JoinError` from a
/// panicking `spawn_blocking` task, poisoned `Session::inner`
/// mutex, 32-bit `u64 → usize` overflow in `EngineConfig::try_from`,
/// or `cera::CeraError::Backend` routed through the `From` impl.
/// Format string matches `cera::CeraError::Backend`'s
/// `"backend: {0}"` — FFI-internal constructors that have already
/// formatted a descriptive message (e.g. "generate_async join
/// error: ...") still read cleanly with the `backend:` label.
///
/// Field is named `detail` rather than `message` for the same
/// `Throwable.message` collision reason as [`FfiError::Io`].
final class FfiErrorBackend extends FfiError {
  const FfiErrorBackend({
    required this.detail,
  });
  final String detail;

  @override
  String toString() {
    return 'FfiErrorBackend(detail: $detail)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorBackend && detail == other.detail;

  @override
  int get hashCode => detail.hashCode;
}

/// The GBNF grammar string passed in `GenerateOpts.grammar` failed to
/// compile. Grammar compilation happens in the FFI wrapper (the compiled
/// grammar object can't cross the boundary, so callers pass the source text
/// and it's parsed here). `detail` carries the parser's diagnostic.
final class FfiErrorGrammarParse extends FfiError {
  const FfiErrorGrammarParse({
    required this.detail,
  });
  final String detail;

  @override
  String toString() {
    return 'FfiErrorGrammarParse(detail: $detail)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorGrammarParse && detail == other.detail;

  @override
  int get hashCode => detail.hashCode;
}

/// A token id passed to `hidden_states_for_tokens` (or another
/// token-taking method) was `>= vocab_size`. Returned as a typed error
/// rather than tripping the model-layer `assert!` (whose panic would
/// unwind through the held session lock and poison it). Mirrors
/// `cera::CeraError::InvalidToken`.
final class FfiErrorInvalidToken extends FfiError {
  const FfiErrorInvalidToken({
    required this.id,
    required this.vocabSize,
  });
  final int id;
  final int vocabSize;

  @override
  String toString() {
    return 'FfiErrorInvalidToken(id: $id, vocabSize: $vocabSize)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorInvalidToken && id == other.id && vocabSize == other.vocabSize;

  @override
  int get hashCode => Object.hash(id, vocabSize);
}

/// A LoRA adapter failed to load ([`LoraAdapters::from_gguf`] /
/// [`LoraAdapters::from_safetensors`]) or was incompatible with the model at
/// attach time (wrong dimensions). `detail` carries the diagnostic.
final class FfiErrorLoraParse extends FfiError {
  const FfiErrorLoraParse({
    required this.detail,
  });
  final String detail;

  @override
  String toString() {
    return 'FfiErrorLoraParse(detail: $detail)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorLoraParse && detail == other.detail;

  @override
  int get hashCode => detail.hashCode;
}

/// A large model/KV allocation could not be satisfied — the device is out of
/// memory for this model at this context size. Returned instead of aborting
/// the process, so a caller can fall back (smaller model or context) or
/// surface a clean error. Mirrors `cera::CeraError::OutOfMemory`.
final class FfiErrorOutOfMemory extends FfiError {
  const FfiErrorOutOfMemory({
    required this.requestedBytes,
  });
  final int requestedBytes;

  @override
  String toString() {
    return 'FfiErrorOutOfMemory(requestedBytes: $requestedBytes)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorOutOfMemory && requestedBytes == other.requestedBytes;

  @override
  int get hashCode => requestedBytes.hashCode;
}

/// A backend's KV-cache compression mode is fixed by the first session that
/// configures it — the compressed and uncompressed caches have different
/// buffer layouts (and the uncompressed one is f32 on CPU/wgpu but f16 on
/// Metal), so only the configured one is ever allocated. Two sessions wanting
/// different modes need two `CeraModel` instances. Mirrors
/// `cera::CeraError::KvCompressionConflict`.
final class FfiErrorKvCompressionConflict extends FfiError {
  const FfiErrorKvCompressionConflict({
    required this.configured,
    required this.requested,
  });
  final String configured;
  final String requested;

  @override
  String toString() {
    return 'FfiErrorKvCompressionConflict(configured: $configured, requested: $requested)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorKvCompressionConflict && configured == other.configured && requested == other.requested;

  @override
  int get hashCode => Object.hash(configured, requested);
}

/// The adapter fits the model, but the active backend has no hook for
/// something it adapts. Mirrors [`cera::CeraError::LoraUnsupportedByBackend`].
///
/// Separate from [`FfiError::LoraParse`] because the two need different
/// handling on the foreign side: `LoraParse` means the adapter or the model
/// pairing is wrong, while this one means only the backend is, so a caller
/// can retry on CPU instead of surfacing "bad adapter" to a user. Today the
/// case is a routed feed-forward (mixture-of-experts) delta on a GPU
/// backend.
///
/// **Appended, not grouped next to `LoraParse`.** UniFFI serializes this
/// enum by ordinal, and the committed Kotlin/Swift/Dart bindings decode it
/// the same way, so inserting mid-enum renumbers every later variant and a
/// prebuilt consumer would decode this one as whatever now holds its old
/// ordinal. New variants go at the end.
final class FfiErrorLoraUnsupportedByBackend extends FfiError {
  const FfiErrorLoraUnsupportedByBackend({
    required this.detail,
  });
  final String detail;

  @override
  String toString() {
    return 'FfiErrorLoraUnsupportedByBackend(detail: $detail)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiErrorLoraUnsupportedByBackend && detail == other.detail;

  @override
  int get hashCode => detail.hashCode;
}

/// A speech boundary event emitted during streaming audio processing.
sealed class FfiVadEvent {
  const FfiVadEvent();
}

final class FfiVadEventSpeechStart extends FfiVadEvent {
  const FfiVadEventSpeechStart({
    required this.sample,
    required this.ms,
  });
  final int sample;
  final double ms;

  @override
  String toString() {
    return 'FfiVadEventSpeechStart(sample: $sample, ms: $ms)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiVadEventSpeechStart && sample == other.sample && ms == other.ms;

  @override
  int get hashCode => Object.hash(sample, ms);
}

final class FfiVadEventSpeechEnd extends FfiVadEvent {
  const FfiVadEventSpeechEnd({
    required this.startSample,
    required this.endSample,
    required this.startMs,
    required this.endMs,
  });
  final int startSample;
  final int endSample;
  final double startMs;
  final double endMs;

  @override
  String toString() {
    return 'FfiVadEventSpeechEnd(startSample: $startSample, endSample: $endSample, startMs: $startMs, endMs: $endMs)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FfiVadEventSpeechEnd && startSample == other.startSample && endSample == other.endSample && startMs == other.startMs && endMs == other.endMs;

  @override
  int get hashCode => Object.hash(startSample, endSample, startMs, endMs);
}

/// Audio sample rate supported by Silero VAD.
enum FfiVadSampleRate {
  rate16kHz,
  rate8kHz,
}

/// Why a decode loop exited. Mirrors [`cera::FinishReason`].
sealed class FinishReason {
  const FinishReason();
}

final class FinishReasonMaxTokens extends FinishReason {
  const FinishReasonMaxTokens();

  @override
  String toString() {
    return 'FinishReasonMaxTokens()';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FinishReasonMaxTokens;

  @override
  int get hashCode => runtimeType.hashCode;
}

final class FinishReasonStop extends FinishReason {
  const FinishReasonStop();

  @override
  String toString() {
    return 'FinishReasonStop()';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FinishReasonStop;

  @override
  int get hashCode => runtimeType.hashCode;
}

final class FinishReasonCancelled extends FinishReason {
  const FinishReasonCancelled();

  @override
  String toString() {
    return 'FinishReasonCancelled()';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FinishReasonCancelled;

  @override
  int get hashCode => runtimeType.hashCode;
}

final class FinishReasonContextFull extends FinishReason {
  const FinishReasonContextFull();

  @override
  String toString() {
    return 'FinishReasonContextFull()';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FinishReasonContextFull;

  @override
  int get hashCode => runtimeType.hashCode;
}

/// A grammar constraint left no token allowed at this step — decoding
/// stopped because the grammar dead-ended. Only reachable when
/// `GenerateOpts.grammar` is set.
final class FinishReasonGrammarDeadEnd extends FinishReason {
  const FinishReasonGrammarDeadEnd();

  @override
  String toString() {
    return 'FinishReasonGrammarDeadEnd()';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FinishReasonGrammarDeadEnd;

  @override
  int get hashCode => runtimeType.hashCode;
}

final class FinishReasonError extends FinishReason {
  const FinishReasonError({
    required this.message,
  });
  final String message;

  @override
  String toString() {
    return 'FinishReasonError(message: $message)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FinishReasonError && message == other.message;

  @override
  int get hashCode => message.hashCode;
}

/// KV-cache compression mode. Mirrors [`cera::kv_cache::KvCompression`].
/// `TurboQuant` is honored by the CPU backend and by both GPU backends (wgpu
/// and native Metal). The GPU paths implement the both-sides mode only: a
/// single-sided (debug) request, or a `head_dim` their kernels can't handle,
/// warns and falls back to that backend's uncompressed KV (f32 on wgpu, f16 on
/// Metal). `F16` is honored by the CPU backend only.
sealed class KvCompression {
  const KvCompression();
}

/// No compression — the backend's uncompressed KV: f32 on CPU and wgpu,
/// f16 on native Metal, whose cache has always been half precision.
final class KvCompressionNone extends KvCompression {
  const KvCompressionNone();

  @override
  String toString() {
    return 'KvCompressionNone()';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is KvCompressionNone;

  @override
  int get hashCode => runtimeType.hashCode;
}

/// f16 KV cache — half-precision keys + values (2 bytes/elem), ~2× less KV
/// bandwidth at decode-at-depth. Near-lossless. CPU LFM2 and
/// dense-transformer paths.
final class KvCompressionF16 extends KvCompression {
  const KvCompressionF16();

  @override
  String toString() {
    return 'KvCompressionF16()';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is KvCompressionF16;

  @override
  int get hashCode => runtimeType.hashCode;
}

/// TurboQuant compression. Both `keys` + `values` true is the
/// production configuration; toggling them individually is
/// primarily for debugging the drift contribution of each side.
/// `seed` drives the per-layer randomized Hadamard rotations.
final class KvCompressionTurboQuant extends KvCompression {
  const KvCompressionTurboQuant({
    required this.seed,
    required this.keys,
    required this.values,
  });
  final int seed;
  final bool keys;
  final bool values;

  @override
  String toString() {
    return 'KvCompressionTurboQuant(seed: $seed, keys: $keys, values: $values)';
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is KvCompressionTurboQuant && seed == other.seed && keys == other.keys && values == other.values;

  @override
  int get hashCode => Object.hash(seed, keys, values);
}

/// The tool-call wire format a model family uses. Mirrors
/// [`cera::tools::ToolFormat`]. Get one from
/// [`CeraEngine::tool_format`] (auto-detected from the model) or set it
/// explicitly.
enum ToolFormat {
  /// LFM2 / LFM2.5: Pythonic `[get_weather(city="Paris")]` in
  /// `<|tool_call_start|>…<|tool_call_end|>`.
  lfm2Pythonic,
  /// Hermes / Qwen: JSON `{"name":…,"arguments":{…}}` in
  /// `<tool_call>…</tool_call>`.
  hermes,
}

/// Typed error surface for `cera-ffi`. Mirrors [`cera::CeraError`] one-
/// to-one so foreign callers can pattern-match on error class (Kotlin
/// `when`, Swift `switch`, Python `match`) instead of string-sniffing
/// a generic message.
///
/// `Backend` is **not** a silent fallback for unmapped `cera::CeraError`
/// variants — the `From<CeraError>` impl is exhaustive, so adding a
/// new cera variant breaks compilation here. `Backend` exists solely
/// for FFI-internal errors that have no cera analog: `JoinError` from
/// a panicking `spawn_blocking` task, a poisoned `Session::inner`
/// mutex, 32-bit `u64 → usize` overflow in `EngineConfig::try_from`.
///
/// Every variant carries the data needed to act on it:
/// `ContextOverflow` exposes `max_seq_len` and `by` so callers can
/// reset or truncate rather than re-reading the message;
/// `UnsupportedInferenceType` exposes the offending value;
/// `Io` preserves the underlying OS error message as a string since
/// `io::Error` isn't UniFFI-marshallable.
///
/// `#[error(...)]` format strings match `cera::CeraError` exactly for
/// every shared variant, so `Display` output is identical whether the
/// error originates from cera directly or routes through the FFI
/// wrapper. Pinned by `ffi_error_display_matches_cera_error_for_every_shared_variant`.
sealed class FfiErrorException implements Exception {
  const FfiErrorException();
}

/// The loaded model doesn't support the modality the caller
/// requested (e.g. `append_audio` on a text-only LLM).
final class FfiErrorExceptionUnsupportedModality extends FfiErrorException {
  const FfiErrorExceptionUnsupportedModality();

  @override
  String toString() {
    return 'FfiErrorExceptionUnsupportedModality()';
  }
}

/// The manifest's `inference_type` is one cera doesn't recognize
/// at this version. Field carries the offending string.
final class FfiErrorExceptionUnsupportedInferenceType extends FfiErrorException {
  const FfiErrorExceptionUnsupportedInferenceType({
    required this.inferenceType,
  });
  final String inferenceType;

  @override
  String toString() {
    return 'FfiErrorExceptionUnsupportedInferenceType(inferenceType: $inferenceType)';
  }
}

/// A concurrent `generate*` call is already in flight on this
/// session. Rust side guards with a mutex; this surfaces when the
/// FFI detects contention.
final class FfiErrorExceptionBusy extends FfiErrorException {
  const FfiErrorExceptionBusy();

  @override
  String toString() {
    return 'FfiErrorExceptionBusy()';
  }
}

/// The caller (or the cancel-on-drop guard) flipped the cancel
/// atomic mid-call. Surfaces from `append_text`, `append_tokens`,
/// and `append_audio` when chunked prefill detects the cancel
/// flag between micro-batches and aborts (see
/// [`cera::Session::append_tokens`] for the chunked-prefill
/// mechanism). Call [`Session::clear_cancel`] to reset the flag
/// so the next call can proceed.
///
/// `generate` reports cancellation via a different path: the
/// call still returns `Ok` with a [`GenerateOutput`] whose
/// `finish_reason` is set to `Cancelled`. Two paths because
/// chunked prefill has nothing useful to return on cancel (no
/// decoded tokens) while decode has accumulated tokens worth
/// preserving.
final class FfiErrorExceptionCancelled extends FfiErrorException {
  const FfiErrorExceptionCancelled();

  @override
  String toString() {
    return 'FfiErrorExceptionCancelled()';
  }
}

/// The context window is full and the session can't shift to make
/// room (e.g. `n_keep == 0`, TurboQuant caches, or the active
/// model doesn't support rope-shift). `max_seq_len` is the cap
/// that was hit; `by` is the overshoot in tokens.
final class FfiErrorExceptionContextOverflow extends FfiErrorException {
  const FfiErrorExceptionContextOverflow({
    required this.maxSeqLen,
    required this.by,
  });
  final int maxSeqLen;
  final int by;

  @override
  String toString() {
    return 'FfiErrorExceptionContextOverflow(maxSeqLen: $maxSeqLen, by: $by)';
  }
}

/// Input buffer was empty (e.g. `append_text("")`, or decode with
/// no prefill state).
final class FfiErrorExceptionEmptyInput extends FfiErrorException {
  const FfiErrorExceptionEmptyInput();

  @override
  String toString() {
    return 'FfiErrorExceptionEmptyInput()';
  }
}

/// Filesystem / mmap / network error surfaced from cera. The
/// underlying `io::Error` isn't marshallable, so the message is
/// flattened to a string. Callers that need the raw kind should
/// parse the `detail` field or open an issue to request a typed
/// field.
///
/// Field is named `detail` rather than `message` because UniFFI's
/// 0.31 Kotlin generator emits `class Io(val `message`) : FfiException()`
/// AND `override val message` in the body when the field is literally
/// named `message`, producing a "conflicting declarations" error
/// (the constructor param collides with the inherited
/// `Throwable.message` override). Renaming to `detail` sidesteps
/// the collision.
///
/// Format string matches `cera::CeraError::Io`'s `"io: {0}"` so
/// foreign `.toString()` / `String(describing:)` gives the same
/// output Rust consumers see.
final class FfiErrorExceptionIo extends FfiErrorException {
  const FfiErrorExceptionIo({
    required this.detail,
  });
  final String detail;

  @override
  String toString() {
    return 'FfiErrorExceptionIo(detail: $detail)';
  }
}

/// FFI-internal error with no cera analog: `JoinError` from a
/// panicking `spawn_blocking` task, poisoned `Session::inner`
/// mutex, 32-bit `u64 → usize` overflow in `EngineConfig::try_from`,
/// or `cera::CeraError::Backend` routed through the `From` impl.
/// Format string matches `cera::CeraError::Backend`'s
/// `"backend: {0}"` — FFI-internal constructors that have already
/// formatted a descriptive message (e.g. "generate_async join
/// error: ...") still read cleanly with the `backend:` label.
///
/// Field is named `detail` rather than `message` for the same
/// `Throwable.message` collision reason as [`FfiError::Io`].
final class FfiErrorExceptionBackend extends FfiErrorException {
  const FfiErrorExceptionBackend({
    required this.detail,
  });
  final String detail;

  @override
  String toString() {
    return 'FfiErrorExceptionBackend(detail: $detail)';
  }
}

/// The GBNF grammar string passed in `GenerateOpts.grammar` failed to
/// compile. Grammar compilation happens in the FFI wrapper (the compiled
/// grammar object can't cross the boundary, so callers pass the source text
/// and it's parsed here). `detail` carries the parser's diagnostic.
final class FfiErrorExceptionGrammarParse extends FfiErrorException {
  const FfiErrorExceptionGrammarParse({
    required this.detail,
  });
  final String detail;

  @override
  String toString() {
    return 'FfiErrorExceptionGrammarParse(detail: $detail)';
  }
}

/// A token id passed to `hidden_states_for_tokens` (or another
/// token-taking method) was `>= vocab_size`. Returned as a typed error
/// rather than tripping the model-layer `assert!` (whose panic would
/// unwind through the held session lock and poison it). Mirrors
/// `cera::CeraError::InvalidToken`.
final class FfiErrorExceptionInvalidToken extends FfiErrorException {
  const FfiErrorExceptionInvalidToken({
    required this.id,
    required this.vocabSize,
  });
  final int id;
  final int vocabSize;

  @override
  String toString() {
    return 'FfiErrorExceptionInvalidToken(id: $id, vocabSize: $vocabSize)';
  }
}

/// A LoRA adapter failed to load ([`LoraAdapters::from_gguf`] /
/// [`LoraAdapters::from_safetensors`]) or was incompatible with the model at
/// attach time (wrong dimensions). `detail` carries the diagnostic.
final class FfiErrorExceptionLoraParse extends FfiErrorException {
  const FfiErrorExceptionLoraParse({
    required this.detail,
  });
  final String detail;

  @override
  String toString() {
    return 'FfiErrorExceptionLoraParse(detail: $detail)';
  }
}

/// A large model/KV allocation could not be satisfied — the device is out of
/// memory for this model at this context size. Returned instead of aborting
/// the process, so a caller can fall back (smaller model or context) or
/// surface a clean error. Mirrors `cera::CeraError::OutOfMemory`.
final class FfiErrorExceptionOutOfMemory extends FfiErrorException {
  const FfiErrorExceptionOutOfMemory({
    required this.requestedBytes,
  });
  final int requestedBytes;

  @override
  String toString() {
    return 'FfiErrorExceptionOutOfMemory(requestedBytes: $requestedBytes)';
  }
}

/// A backend's KV-cache compression mode is fixed by the first session that
/// configures it — the compressed and uncompressed caches have different
/// buffer layouts (and the uncompressed one is f32 on CPU/wgpu but f16 on
/// Metal), so only the configured one is ever allocated. Two sessions wanting
/// different modes need two `CeraModel` instances. Mirrors
/// `cera::CeraError::KvCompressionConflict`.
final class FfiErrorExceptionKvCompressionConflict extends FfiErrorException {
  const FfiErrorExceptionKvCompressionConflict({
    required this.configured,
    required this.requested,
  });
  final String configured;
  final String requested;

  @override
  String toString() {
    return 'FfiErrorExceptionKvCompressionConflict(configured: $configured, requested: $requested)';
  }
}

/// The adapter fits the model, but the active backend has no hook for
/// something it adapts. Mirrors [`cera::CeraError::LoraUnsupportedByBackend`].
///
/// Separate from [`FfiError::LoraParse`] because the two need different
/// handling on the foreign side: `LoraParse` means the adapter or the model
/// pairing is wrong, while this one means only the backend is, so a caller
/// can retry on CPU instead of surfacing "bad adapter" to a user. Today the
/// case is a routed feed-forward (mixture-of-experts) delta on a GPU
/// backend.
///
/// **Appended, not grouped next to `LoraParse`.** UniFFI serializes this
/// enum by ordinal, and the committed Kotlin/Swift/Dart bindings decode it
/// the same way, so inserting mid-enum renumbers every later variant and a
/// prebuilt consumer would decode this one as whatever now holds its old
/// ordinal. New variants go at the end.
final class FfiErrorExceptionLoraUnsupportedByBackend extends FfiErrorException {
  const FfiErrorExceptionLoraUnsupportedByBackend({
    required this.detail,
  });
  final String detail;

  @override
  String toString() {
    return 'FfiErrorExceptionLoraUnsupportedByBackend(detail: $detail)';
  }
}

String _encodeBackendPreference(BackendPreference value) {
  return switch (value) {
    BackendPreference.auto => 'auto',
    BackendPreference.cpu => 'cpu',
    BackendPreference.gpu => 'gpu',
    BackendPreference.metal => 'metal',
  };
}

BackendPreference _decodeBackendPreference(String raw) {
  return switch (raw) {
    'auto' => BackendPreference.auto,
    'cpu' => BackendPreference.cpu,
    'gpu' => BackendPreference.gpu,
    'metal' => BackendPreference.metal,
    _ => throw StateError('Unknown BackendPreference variant: $raw'),
  };
}

String _encodeFfiError(FfiError value) {
  if (value is FfiErrorUnsupportedModality) {
    return jsonEncode({
      'tag': 'unsupportedModality',
    });
  }
  if (value is FfiErrorUnsupportedInferenceType) {
    return jsonEncode({
      'tag': 'unsupportedInferenceType',
      'inferenceType': value.inferenceType,
    });
  }
  if (value is FfiErrorBusy) {
    return jsonEncode({
      'tag': 'busy',
    });
  }
  if (value is FfiErrorCancelled) {
    return jsonEncode({
      'tag': 'cancelled',
    });
  }
  if (value is FfiErrorContextOverflow) {
    return jsonEncode({
      'tag': 'contextOverflow',
      'maxSeqLen': value.maxSeqLen,
      'by': value.by,
    });
  }
  if (value is FfiErrorEmptyInput) {
    return jsonEncode({
      'tag': 'emptyInput',
    });
  }
  if (value is FfiErrorIo) {
    return jsonEncode({
      'tag': 'io',
      'detail': value.detail,
    });
  }
  if (value is FfiErrorBackend) {
    return jsonEncode({
      'tag': 'backend',
      'detail': value.detail,
    });
  }
  if (value is FfiErrorGrammarParse) {
    return jsonEncode({
      'tag': 'grammarParse',
      'detail': value.detail,
    });
  }
  if (value is FfiErrorInvalidToken) {
    return jsonEncode({
      'tag': 'invalidToken',
      'id': value.id,
      'vocabSize': value.vocabSize,
    });
  }
  if (value is FfiErrorLoraParse) {
    return jsonEncode({
      'tag': 'loraParse',
      'detail': value.detail,
    });
  }
  if (value is FfiErrorOutOfMemory) {
    return jsonEncode({
      'tag': 'outOfMemory',
      'requestedBytes': value.requestedBytes,
    });
  }
  if (value is FfiErrorKvCompressionConflict) {
    return jsonEncode({
      'tag': 'kvCompressionConflict',
      'configured': value.configured,
      'requested': value.requested,
    });
  }
  if (value is FfiErrorLoraUnsupportedByBackend) {
    return jsonEncode({
      'tag': 'loraUnsupportedByBackend',
      'detail': value.detail,
    });
  }
  throw StateError('Unknown FfiError variant instance: $value');
}

FfiError _decodeFfiError(String raw) {
  final Map<String, dynamic> map = jsonDecode(raw) as Map<String, dynamic>;
  final String? tag = map['tag'] as String?;
  switch (tag) {
    case 'unsupportedModality':
      return FfiErrorUnsupportedModality(
      );
    case 'unsupportedInferenceType':
      return FfiErrorUnsupportedInferenceType(
        inferenceType: map['inferenceType'] as String,
      );
    case 'busy':
      return FfiErrorBusy(
      );
    case 'cancelled':
      return FfiErrorCancelled(
      );
    case 'contextOverflow':
      return FfiErrorContextOverflow(
        maxSeqLen: (map['maxSeqLen'] as num).toInt(),
        by: (map['by'] as num).toInt(),
      );
    case 'emptyInput':
      return FfiErrorEmptyInput(
      );
    case 'io':
      return FfiErrorIo(
        detail: map['detail'] as String,
      );
    case 'backend':
      return FfiErrorBackend(
        detail: map['detail'] as String,
      );
    case 'grammarParse':
      return FfiErrorGrammarParse(
        detail: map['detail'] as String,
      );
    case 'invalidToken':
      return FfiErrorInvalidToken(
        id: (map['id'] as num).toInt(),
        vocabSize: (map['vocabSize'] as num).toInt(),
      );
    case 'loraParse':
      return FfiErrorLoraParse(
        detail: map['detail'] as String,
      );
    case 'outOfMemory':
      return FfiErrorOutOfMemory(
        requestedBytes: (map['requestedBytes'] as num).toInt(),
      );
    case 'kvCompressionConflict':
      return FfiErrorKvCompressionConflict(
        configured: map['configured'] as String,
        requested: map['requested'] as String,
      );
    case 'loraUnsupportedByBackend':
      return FfiErrorLoraUnsupportedByBackend(
        detail: map['detail'] as String,
      );
    default:
      throw StateError('Unknown FfiError variant tag: $tag');
  }
}

String _encodeFfiVadEvent(FfiVadEvent value) {
  if (value is FfiVadEventSpeechStart) {
    return jsonEncode({
      'tag': 'speechStart',
      'sample': value.sample,
      'ms': value.ms,
    });
  }
  if (value is FfiVadEventSpeechEnd) {
    return jsonEncode({
      'tag': 'speechEnd',
      'startSample': value.startSample,
      'endSample': value.endSample,
      'startMs': value.startMs,
      'endMs': value.endMs,
    });
  }
  throw StateError('Unknown FfiVadEvent variant instance: $value');
}

FfiVadEvent _decodeFfiVadEvent(String raw) {
  final Map<String, dynamic> map = jsonDecode(raw) as Map<String, dynamic>;
  final String? tag = map['tag'] as String?;
  switch (tag) {
    case 'speechStart':
      return FfiVadEventSpeechStart(
        sample: (map['sample'] as num).toInt(),
        ms: (map['ms'] as num).toDouble(),
      );
    case 'speechEnd':
      return FfiVadEventSpeechEnd(
        startSample: (map['startSample'] as num).toInt(),
        endSample: (map['endSample'] as num).toInt(),
        startMs: (map['startMs'] as num).toDouble(),
        endMs: (map['endMs'] as num).toDouble(),
      );
    default:
      throw StateError('Unknown FfiVadEvent variant tag: $tag');
  }
}

String _encodeFfiVadSampleRate(FfiVadSampleRate value) {
  return switch (value) {
    FfiVadSampleRate.rate16kHz => 'rate16kHz',
    FfiVadSampleRate.rate8kHz => 'rate8kHz',
  };
}

FfiVadSampleRate _decodeFfiVadSampleRate(String raw) {
  return switch (raw) {
    'rate16kHz' => FfiVadSampleRate.rate16kHz,
    'rate8kHz' => FfiVadSampleRate.rate8kHz,
    _ => throw StateError('Unknown FfiVadSampleRate variant: $raw'),
  };
}

String _encodeFinishReason(FinishReason value) {
  if (value is FinishReasonMaxTokens) {
    return jsonEncode({
      'tag': 'maxTokens',
    });
  }
  if (value is FinishReasonStop) {
    return jsonEncode({
      'tag': 'stop',
    });
  }
  if (value is FinishReasonCancelled) {
    return jsonEncode({
      'tag': 'cancelled',
    });
  }
  if (value is FinishReasonContextFull) {
    return jsonEncode({
      'tag': 'contextFull',
    });
  }
  if (value is FinishReasonGrammarDeadEnd) {
    return jsonEncode({
      'tag': 'grammarDeadEnd',
    });
  }
  if (value is FinishReasonError) {
    return jsonEncode({
      'tag': 'error',
      'message': value.message,
    });
  }
  throw StateError('Unknown FinishReason variant instance: $value');
}

FinishReason _decodeFinishReason(String raw) {
  final Map<String, dynamic> map = jsonDecode(raw) as Map<String, dynamic>;
  final String? tag = map['tag'] as String?;
  switch (tag) {
    case 'maxTokens':
      return FinishReasonMaxTokens(
      );
    case 'stop':
      return FinishReasonStop(
      );
    case 'cancelled':
      return FinishReasonCancelled(
      );
    case 'contextFull':
      return FinishReasonContextFull(
      );
    case 'grammarDeadEnd':
      return FinishReasonGrammarDeadEnd(
      );
    case 'error':
      return FinishReasonError(
        message: map['message'] as String,
      );
    default:
      throw StateError('Unknown FinishReason variant tag: $tag');
  }
}

String _encodeKvCompression(KvCompression value) {
  if (value is KvCompressionNone) {
    return jsonEncode({
      'tag': 'none',
    });
  }
  if (value is KvCompressionF16) {
    return jsonEncode({
      'tag': 'f16',
    });
  }
  if (value is KvCompressionTurboQuant) {
    return jsonEncode({
      'tag': 'turboQuant',
      'seed': value.seed,
      'keys': value.keys,
      'values': value.values,
    });
  }
  throw StateError('Unknown KvCompression variant instance: $value');
}

KvCompression _decodeKvCompression(String raw) {
  final Map<String, dynamic> map = jsonDecode(raw) as Map<String, dynamic>;
  final String? tag = map['tag'] as String?;
  switch (tag) {
    case 'none':
      return KvCompressionNone(
      );
    case 'f16':
      return KvCompressionF16(
      );
    case 'turboQuant':
      return KvCompressionTurboQuant(
        seed: (map['seed'] as num).toInt(),
        keys: map['keys'] as bool,
        values: map['values'] as bool,
      );
    default:
      throw StateError('Unknown KvCompression variant tag: $tag');
  }
}

String _encodeToolFormat(ToolFormat value) {
  return switch (value) {
    ToolFormat.lfm2Pythonic => 'lfm2Pythonic',
    ToolFormat.hermes => 'hermes',
  };
}

ToolFormat _decodeToolFormat(String raw) {
  return switch (raw) {
    'lfm2Pythonic' => ToolFormat.lfm2Pythonic,
    'hermes' => ToolFormat.hermes,
    _ => throw StateError('Unknown ToolFormat variant: $raw'),
  };
}

String _encodeFfiErrorException(FfiErrorException value) {
  if (value is FfiErrorExceptionUnsupportedModality) {
    return jsonEncode({
      'tag': 'unsupportedModality',
    });
  }
  if (value is FfiErrorExceptionUnsupportedInferenceType) {
    return jsonEncode({
      'tag': 'unsupportedInferenceType',
      'inferenceType': value.inferenceType,
    });
  }
  if (value is FfiErrorExceptionBusy) {
    return jsonEncode({
      'tag': 'busy',
    });
  }
  if (value is FfiErrorExceptionCancelled) {
    return jsonEncode({
      'tag': 'cancelled',
    });
  }
  if (value is FfiErrorExceptionContextOverflow) {
    return jsonEncode({
      'tag': 'contextOverflow',
      'maxSeqLen': value.maxSeqLen,
      'by': value.by,
    });
  }
  if (value is FfiErrorExceptionEmptyInput) {
    return jsonEncode({
      'tag': 'emptyInput',
    });
  }
  if (value is FfiErrorExceptionIo) {
    return jsonEncode({
      'tag': 'io',
      'detail': value.detail,
    });
  }
  if (value is FfiErrorExceptionBackend) {
    return jsonEncode({
      'tag': 'backend',
      'detail': value.detail,
    });
  }
  if (value is FfiErrorExceptionGrammarParse) {
    return jsonEncode({
      'tag': 'grammarParse',
      'detail': value.detail,
    });
  }
  if (value is FfiErrorExceptionInvalidToken) {
    return jsonEncode({
      'tag': 'invalidToken',
      'id': value.id,
      'vocabSize': value.vocabSize,
    });
  }
  if (value is FfiErrorExceptionLoraParse) {
    return jsonEncode({
      'tag': 'loraParse',
      'detail': value.detail,
    });
  }
  if (value is FfiErrorExceptionOutOfMemory) {
    return jsonEncode({
      'tag': 'outOfMemory',
      'requestedBytes': value.requestedBytes,
    });
  }
  if (value is FfiErrorExceptionKvCompressionConflict) {
    return jsonEncode({
      'tag': 'kvCompressionConflict',
      'configured': value.configured,
      'requested': value.requested,
    });
  }
  if (value is FfiErrorExceptionLoraUnsupportedByBackend) {
    return jsonEncode({
      'tag': 'loraUnsupportedByBackend',
      'detail': value.detail,
    });
  }
  throw StateError('Unknown FfiErrorException exception instance: $value');
}

FfiErrorException _decodeFfiErrorException(Object? raw) {
  final Map<String, dynamic> map = raw is String ? (jsonDecode(raw) as Map<String, dynamic>) : (raw as Map<String, dynamic>);
  final String? tag = map['tag'] as String?;
  switch (tag) {
    case 'unsupportedModality':
      return const FfiErrorExceptionUnsupportedModality();
    case 'unsupportedInferenceType':
      return FfiErrorExceptionUnsupportedInferenceType(
        inferenceType: map['inferenceType'] as String,
      );
    case 'busy':
      return const FfiErrorExceptionBusy();
    case 'cancelled':
      return const FfiErrorExceptionCancelled();
    case 'contextOverflow':
      return FfiErrorExceptionContextOverflow(
        maxSeqLen: (map['maxSeqLen'] as num).toInt(),
        by: (map['by'] as num).toInt(),
      );
    case 'emptyInput':
      return const FfiErrorExceptionEmptyInput();
    case 'io':
      return FfiErrorExceptionIo(
        detail: map['detail'] as String,
      );
    case 'backend':
      return FfiErrorExceptionBackend(
        detail: map['detail'] as String,
      );
    case 'grammarParse':
      return FfiErrorExceptionGrammarParse(
        detail: map['detail'] as String,
      );
    case 'invalidToken':
      return FfiErrorExceptionInvalidToken(
        id: (map['id'] as num).toInt(),
        vocabSize: (map['vocabSize'] as num).toInt(),
      );
    case 'loraParse':
      return FfiErrorExceptionLoraParse(
        detail: map['detail'] as String,
      );
    case 'outOfMemory':
      return FfiErrorExceptionOutOfMemory(
        requestedBytes: (map['requestedBytes'] as num).toInt(),
      );
    case 'kvCompressionConflict':
      return FfiErrorExceptionKvCompressionConflict(
        configured: map['configured'] as String,
        requested: map['requested'] as String,
      );
    case 'loraUnsupportedByBackend':
      return FfiErrorExceptionLoraUnsupportedByBackend(
        detail: map['detail'] as String,
      );
    default:
      throw StateError('Unknown FfiErrorException exception tag: $tag');
  }
}

final class BackendPreferenceFfiCodec {
  const BackendPreferenceFfiCodec._();

  static String encode(BackendPreference value) => _encodeBackendPreference(value);

  static BackendPreference decode(String raw) => _decodeBackendPreference(raw);
}

final class FfiErrorFfiCodec {
  const FfiErrorFfiCodec._();

  static String encode(FfiError value) => _encodeFfiError(value);

  static FfiError decode(String raw) => _decodeFfiError(raw);
}

final class FfiVadEventFfiCodec {
  const FfiVadEventFfiCodec._();

  static String encode(FfiVadEvent value) => _encodeFfiVadEvent(value);

  static FfiVadEvent decode(String raw) => _decodeFfiVadEvent(raw);
}

final class FfiVadSampleRateFfiCodec {
  const FfiVadSampleRateFfiCodec._();

  static String encode(FfiVadSampleRate value) => _encodeFfiVadSampleRate(value);

  static FfiVadSampleRate decode(String raw) => _decodeFfiVadSampleRate(raw);
}

final class FinishReasonFfiCodec {
  const FinishReasonFfiCodec._();

  static String encode(FinishReason value) => _encodeFinishReason(value);

  static FinishReason decode(String raw) => _decodeFinishReason(raw);
}

final class KvCompressionFfiCodec {
  const KvCompressionFfiCodec._();

  static String encode(KvCompression value) => _encodeKvCompression(value);

  static KvCompression decode(String raw) => _decodeKvCompression(raw);
}

final class ToolFormatFfiCodec {
  const ToolFormatFfiCodec._();

  static String encode(ToolFormat value) => _encodeToolFormat(value);

  static ToolFormat decode(String raw) => _decodeToolFormat(raw);
}

final class FfiErrorExceptionFfiCodec {
  const FfiErrorExceptionFfiCodec._();

  static String encode(FfiErrorException value) => _encodeFfiErrorException(value);

  static FfiErrorException decode(Object? raw) => _decodeFfiErrorException(raw);
}


/// Remote model-bundle downloader + on-disk cache. Wraps
/// [`cera::bundle::BundleRepo`]; construct once per application with
/// a persistent `store_dir` and reuse across engine loads so the
/// HTTP client pool + downloaded-file cache are shared.
///
/// On Android the `store_dir` should typically be
/// `Context.getFilesDir()` (persistent), not `getCacheDir()` (OS-
/// purgeable under storage pressure). On iOS / macOS, the app's
/// Application Support or a dedicated subdirectory under Documents
/// is a reasonable baseline.
///
/// Cache layout mirrors the remote URL structure under
/// `<store_dir>/huggingface.co/<full path>`, so inspecting the
/// on-disk state with a file browser is straightforward and multiple
/// cera-powered apps on the same device can share the same cache
/// directory without conflicting.
final class BundleRepo {
  BundleRepo._();

  bool get isClosed => _unsupportedOnWeb('BundleRepo.isClosed');

  void close() => _unsupportedOnWeb('BundleRepo.close');

  /// Create a new repo rooted at `store_dir`. The directory doesn't
  /// need to exist yet — it's created on the first download. Pass
  /// the same path to subsequent runs to reuse the cached bundles.
  static BundleRepo create(String storeDir) => _unsupportedOnWeb('BundleRepo.create');

  /// Create a new repo rooted at `store_dir` with a foreign
  /// [`DownloadProgressSink`] attached. The sink fires periodically
  /// during cache-miss downloads (every ~256 KB written + once at
  /// end-of-stream). Cache-hit resolves don't fire any callbacks.
  /// The same sink receives events for every file the repo
  /// downloads — distinguish per-file progress by the `url`
  /// argument on each callback.
  ///
  /// Construction-time attachment (rather than per-call) matches
  /// how mobile apps drive a single download-progress UI across
  /// multiple files in one logical bundle (manifest + GGUF + …):
  /// one repo, one sink, one progress bar. If you need to tear
  /// down the sink mid-app-lifecycle, drop the repo + construct a
  /// new one — Arc-based, so all in-flight calls finish on the
  /// old sink and new calls go to the new one.
  static BundleRepo withProgress(String storeDir, DownloadProgressSink progress) => _unsupportedOnWeb('BundleRepo.withProgress');

  /// Total bytes currently held in the cache. Returns `0` if the
  /// `store_dir` doesn't exist yet (no downloads have run).
  /// O(n) over the cache contents; for a multi-GB cache it's a
  /// real walk, not a constant-time query — UIs surfacing the
  /// value should run it off the main thread (e.g. via
  /// `withContext(Dispatchers.IO)` on Kotlin or
  /// `Task.detached` on Swift).
  ///
  /// Mobile apps use this to drive a "Storage: X MB used" line in
  /// settings or to gate a "Clear cache" button on actual
  /// non-zero usage.
  int cacheSize() => _unsupportedOnWeb('BundleRepo.cacheSize');

  /// Wipe every file the repo has cached, leaving `store_dir`
  /// itself in place so subsequent downloads land in the same
  /// path. Idempotent — calling on an empty repo or non-existent
  /// `store_dir` is a no-op success.
  ///
  /// Mobile apps trigger this from a "Clear downloaded models"
  /// settings action. Caller is responsible for serializing
  /// against in-flight downloads — typically trivial since the
  /// action is user-driven.
  void clearCache() => _unsupportedOnWeb('BundleRepo.clearCache');

  /// Download all assets for a bundle ID and quantization to the local cache
  /// without loading model weights into memory or creating an engine.
  void downloadBundle(String bundleId, String quant) => _unsupportedOnWeb('BundleRepo.downloadBundle');

  /// The directory this repo caches bundles under. Matches what was
  /// passed to [`BundleRepo::new`] / [`BundleRepo::with_progress`],
  /// useful for log / telemetry.
  String storeDir() => _unsupportedOnWeb('BundleRepo.storeDir');
}

final class BundleRepoFfiCodec {
  static int lower(BundleRepo value) => _unsupportedOnWeb('BundleRepoFfiCodec.lower');
  static BundleRepo lift(int handle) => _unsupportedOnWeb('BundleRepoFfiCodec.lift');
}

/// Owning handle to a loaded model. Mirrors [`cera::CeraEngine`];
/// `#[uniffi::Object]` requires `Arc<Self>` wrapping which matches how
/// the underlying engine is already used internally.
final class CeraEngine {
  CeraEngine._();

  bool get isClosed => _unsupportedOnWeb('CeraEngine.isClosed');

  void close() => _unsupportedOnWeb('CeraEngine.close');

  /// Load a model by LeapBundles ID + quantization selector, e.g.
  /// `from_bundle_id("LFM2-1.2B-GGUF", "Q4_0", config)`. Resolves
  /// to the matching `<bundle_id>/<quant>.json` manifest under
  /// `huggingface.co/LiquidAI/LeapBundles` and downloads whatever
  /// isn't already in `config.bundle_repo`'s on-disk cache.
  ///
  /// `config.bundle_repo` must be set; otherwise this returns an
  /// [`FfiError::Backend`] telling the caller to construct a
  /// [`BundleRepo`] and attach it. Idempotent across calls — the
  /// repo's cache deduplicates subsequent downloads.
  ///
  /// Blocking: this call fetches over the network on first run +
  /// opens / parses the GGUF. Foreign async runtimes should wrap
  /// the call in `spawn_blocking` / its equivalent. (An async
  /// counterpart matching `generate_async` could be added later;
  /// not in this PR.)
  static CeraEngine fromBundleId(String bundleId, String quant, EngineConfig config) => _unsupportedOnWeb('CeraEngine.fromBundleId');

  /// Async variant of [`CeraEngine::from_bundle_id`] — offloads the
  /// manifest + GGUF download and the engine construction onto a
  /// tokio blocking worker so the caller's async context isn't
  /// stalled. Foreign async runtimes (Kotlin coroutines, Swift
  /// `async`, Python `asyncio`) `.await` it directly.
  ///
  /// `config.bundle_repo` must be set (same constraint as the sync
  /// twin); construct a [`BundleRepo`] rooted at a persistent cache
  /// directory and attach it to the config before calling.
  ///
  /// Cancellation semantics (weaker than [`Session::generate_async`]):
  /// dropping the returned future drops the `AbortOnDrop` guard,
  /// which calls `AbortHandle::abort` on the spawned task. That
  /// cancels the task if it's still queued on tokio's blocking
  /// pool, so a not-yet-started download never runs. But if the
  /// task has started, abort is a no-op — the download is a
  /// `reqwest::blocking` call with no cooperative cancel point,
  /// and cera's engine-construction code (tokenizer build, model
  /// load, KV alloc) also isn't interruptible. In that case the
  /// task runs to completion and the engine is constructed then
  /// dropped; the downloaded bundle stays cached, so the caller's
  /// next attempt starts from that cache hit. Bandwidth isn't
  /// wasted, it's just shifted.
  ///
  /// `JoinError` from a panicking blocking closure surfaces as
  /// [`FfiError::Backend`] with a diagnostic prefix, same as
  /// [`Session::generate_async`].
  static Future<CeraEngine> fromBundleIdAsync(String bundleId, String quant, EngineConfig config) => _unsupportedOnWeb('CeraEngine.fromBundleIdAsync');

  /// Load a model from GGUF bytes already in memory.
  ///
  /// For callers with no filesystem to point [`CeraEngine::from_path`]
  /// at: a browser, an encrypted blob decrypted in memory, an asset
  /// read out of an archive. It is the one constructor a WebAssembly
  /// build can also offer, so code written against it ports across.
  ///
  /// **Not a streaming API.** GGUF is random-access: tensor data is
  /// addressed by offset and read throughout inference, so the whole
  /// file has to be resident before the first token. You can download
  /// over a stream, but you must accumulate it all before calling
  /// this. There is no partial-model inference.
  ///
  /// **Prefer [`CeraEngine::from_path`] whenever a path exists.** That
  /// route memory-maps the file, so tensor pages stay owned by the
  /// kernel: shared between processes and evictable under pressure.
  /// These bytes are committed resident memory for as long as the
  /// engine lives, which on a phone is the difference between a model
  /// the OS can page out and one that counts against your footprint.
  /// To load from the network on a platform that has a filesystem,
  /// stream to disk and use `from_path` (which is what [`BundleRepo`]
  /// does), rather than buffering the model here.
  ///
  /// Text-only: the bytes are a bare GGUF with no accompanying
  /// manifest, so there is nothing to point at a vision encoder or an
  /// audio decoder. Multimodal models need
  /// [`CeraEngine::from_parts`], `from_path`, or
  /// [`CeraEngine::from_bundle_id`]. `config.bundle_repo` is ignored.
  static CeraEngine fromBytes(Uint8List bytes, EngineConfig config) => _unsupportedOnWeb('CeraEngine.fromBytes');

  /// Async variant of [`CeraEngine::from_bytes`]: the in-memory twin of
  /// [`CeraEngine::from_path_async`], for callers with no filesystem.
  ///
  /// This one benefits more than the path variant: `from_bytes` has no
  /// mmap to lean on, so every tensor is already resident and the whole
  /// parse plus tokenizer build happens inline. Same weak cancellation.
  ///
  /// The `bytes` are moved into the blocking task, so a dropped future
  /// releases them when the task finishes rather than when it is
  /// dropped.
  static Future<CeraEngine> fromBytesAsync(Uint8List bytes, EngineConfig config) => _unsupportedOnWeb('CeraEngine.fromBytesAsync');

  /// Load a multi-file bundle from memory: the model GGUF plus its
  /// multimodal projector ("mmproj").
  ///
  /// This is the constructor a VL or audio model needs when there is
  /// no filesystem, and [`CeraEngine::from_bytes`] structurally cannot
  /// be: the vision tower and the audio encoder live in a *second*
  /// GGUF, and that one takes a single buffer. Same inputs and same
  /// rules as the wasm build's `fromGgufParts`, so a portable layer
  /// over both has one shape to target.
  ///
  /// `multimodal_projector` may be `None`, which makes this exactly
  /// `from_bytes` with an explicit config.
  ///
  /// **Modality is inferred from the arguments, not just the header.**
  /// Every published LFM2-VL model reports `architecture = "lfm2"`,
  /// the same string a text model reports, because the vision half is
  /// entirely in the mmproj. So supplying one alongside a text-arch
  /// model is taken as the statement of intent it is and loads as
  /// image-to-text; audio models already identify themselves and are
  /// unaffected. Pass `inference_type` explicitly to override
  /// (`"llama.cpp/text-to-text"`, `"llama.cpp/image-to-text"`,
  /// `"llama.cpp/lfm2-audio-v1"`).
  ///
  /// A malformed or mismatched mmproj is **not** fatal: it warns, and
  /// the bundle still serves text with `capabilities().image_in`
  /// staying false. That mirrors the path-based loaders rather than
  /// failing a whole load over a sidecar.
  ///
  /// **Prefer `from_path` whenever a path exists**, for the same
  /// memory reason as [`CeraEngine::from_bytes`]: these buffers are
  /// committed resident memory for the engine's lifetime, and a VL
  /// bundle is the model *plus* the tower. `config.bundle_repo` is
  /// ignored.
  static CeraEngine fromParts(Uint8List bytes, Uint8List? multimodalProjector, String? inferenceType, EngineConfig config) => _unsupportedOnWeb('CeraEngine.fromParts');

  /// Async variant of [`CeraEngine::from_parts`].
  ///
  /// Wanted more than the text-only twin, not less: a VL bundle is the
  /// model *plus* its tower, so there is strictly more parsing to keep
  /// off the caller's thread, and the vision encoder's weights are
  /// built during the load. Same weak cancellation, and both buffers
  /// are moved into the blocking task.
  static Future<CeraEngine> fromPartsAsync(Uint8List bytes, Uint8List? multimodalProjector, String? inferenceType, EngineConfig config) => _unsupportedOnWeb('CeraEngine.fromPartsAsync');

  /// Load a model from a local filesystem path. Accepts the same
  /// inputs as the native [`cera::CeraEngine::from_path`]: a bare
  /// `.gguf`, a LeapBundles `.json` manifest, or a directory
  /// containing exactly one `.json` manifest.
  ///
  /// If the manifest carries `http(s)://` URLs for its files,
  /// `config.bundle_repo` must be set — otherwise those URLs fail
  /// to resolve. For a pure-local workflow (bundle already on
  /// disk) leave `bundle_repo = None`.
  static CeraEngine fromPath(String path, EngineConfig config) => _unsupportedOnWeb('CeraEngine.fromPath');

  /// Async variant of [`CeraEngine::from_path`]: moves the GGUF open,
  /// tokenizer build, and KV allocation onto a tokio blocking worker.
  ///
  /// The sync twin is not cheap enough to call from a UI thread. GGUF
  /// tensor data is memory-mapped rather than read, so the cost is not
  /// proportional to file size, but the tokenizer is built eagerly and
  /// a large vocabulary's merge table is real work: enough to drop
  /// frames, and on a cold page cache the metadata reads are disk-bound
  /// on top. Foreign UI code should prefer this everywhere.
  ///
  /// Cancellation is the weak form documented on
  /// [`CeraEngine::from_bundle_id_async`]: dropping the future aborts
  /// the task only while it is still queued. Engine construction has no
  /// cooperative cancel point, so once started it runs to completion and
  /// the result is dropped.
  static Future<CeraEngine> fromPathAsync(String path, EngineConfig config) => _unsupportedOnWeb('CeraEngine.fromPathAsync');

  /// Render the model's chat template against a sequence of
  /// `ChatMessage`s. `add_generation_prompt = true` appends the
  /// model's "now it's the assistant's turn" suffix (typical when
  /// driving an interactive chat); `false` produces a transcript
  /// the model can keep continuing.
  ///
  /// Returns [`FfiError::Backend`] if the model has no chat
  /// template (check [`CeraEngine::has_chat_template`] first) or
  /// if the template fails to render against the supplied messages.
  String applyChatTemplate(List<ChatMessage> messages, bool addGenerationPrompt) => _unsupportedOnWeb('CeraEngine.applyChatTemplate');

  /// Like [`CeraEngine::apply_chat_template`], but also passes a `tools`
  /// array so a tool-trained model renders its tool-definition block. Pass an
  /// empty `tools` for identical behavior to the plain call.
  String applyChatTemplateWithTools(List<ChatMessage> messages, List<ToolDef> tools, bool addGenerationPrompt) => _unsupportedOnWeb('CeraEngine.applyChatTemplateWithTools');

  /// Beginning-of-sequence token ID, if the model has one.
  /// LLaMA-family models typically do; some don't. Honor
  /// [`ModelMetadata::add_bos_token`] when deciding whether to
  /// prepend it manually to a prompt.
  int? bosToken() => _unsupportedOnWeb('CeraEngine.bosToken');

  /// What this model accepts as input / emits as output. Derived at
  /// load time from the manifest's `inference_type`.
  ModalityCapabilities capabilities() => _unsupportedOnWeb('CeraEngine.capabilities');

  /// Clear the engine's in-memory warm KV prefix cache, preserving on-disk cold cache.
  /// Call this from host OS memory pressure warnings (e.g. iOS `applicationDidReceiveMemoryWarning`
  /// or Android `onTrimMemory`) to immediately free RAM without losing persistent cached prefixes.
  void clearPrefixCache() => _unsupportedOnWeb('CeraEngine.clearPrefixCache');

  /// Resolved context-window size (KV cache cap) the engine was
  /// configured with. Mirrors the `context_size` field of the
  /// [`EngineConfig`] passed to `from_path` / `from_bundle_id`,
  /// with the `0` → `model.max_seq_len` defaulting already
  /// applied so callers always see a meaningful number rather
  /// than the internal `usize::MAX` sentinel.
  ///
  /// Note this is the **engine-level** requested cap, not a
  /// per-session ceiling. cera core clamps the model's
  /// `max_seq_len` at load time to `min(requested_context,
  /// gguf_max_seq_len)` (see `cera/src/model/lfm2.rs`), so
  /// [`Self::metadata`]`.max_seq_len` is already the effective
  /// ceiling for any session built from this engine — `context_size`
  /// is informational ("what cap did this engine load with?")
  /// rather than a value callers should `min(...)` against.
  int contextSize() => _unsupportedOnWeb('CeraEngine.contextSize');

  /// Decode token IDs back to text. Out-of-vocab IDs are silently
  /// skipped (omitted from the decoded output) — `BpeTokenizer::decode`
  /// only appends bytes for IDs it has in `vocab.get(id)`. No
  /// substitution glyph, no error. Callers that want to detect
  /// invalid IDs should validate against `vocab_size()` first.
  String decodeTokens(List<int> tokens) => _unsupportedOnWeb('CeraEngine.decodeTokens');

  /// Returns default `GenerateOpts` for this engine, pre-populated with
  /// advisory sampling defaults from the bundle manifest (if any) or standard defaults.
  GenerateOpts defaultGenerateOpts() => _unsupportedOnWeb('CeraEngine.defaultGenerateOpts');

  /// Detect PII entity spans in text using the loaded token classification model.
  List<FfiEntitySpan> detectPii(String text) => _unsupportedOnWeb('CeraEngine.detectPii');

  /// Encode `text` into token IDs using the model's BPE tokenizer.
  /// Empty input returns an empty vec.
  List<int> encodeText(String text) => _unsupportedOnWeb('CeraEngine.encodeText');

  /// Encode `text` with optional special markers — the analog of llama.cpp's
  /// `llama_tokenize(..., add_special)`. When `add_special` is true, BOS is
  /// prepended iff the GGUF declares `tokenizer.ggml.add_bos_token` and EOS
  /// appended iff it declares `tokenizer.ggml.add_eos_token`, so token counts
  /// match llama.cpp for the same text (benchmark parity). With
  /// `add_special = false` this is exactly [`Self::encode_text`]. Prefer this
  /// over hand-prepending BOS via [`ModelMetadata::add_bos_token`].
  List<int> encodeTextSpecial(String text, bool addSpecial) => _unsupportedOnWeb('CeraEngine.encodeTextSpecial');

  /// End-of-sequence / end-of-text token ID, if the model has one.
  /// Used as a default stop-token by the sampler; callers can also
  /// pass it explicitly in [`GenerateOpts::stop_tokens`].
  int? eosToken() => _unsupportedOnWeb('CeraEngine.eosToken');

  /// `true` if the model's tokenizer carries a chat template (a
  /// minijinja string from GGUF metadata). Foreign callers should
  /// check this before calling [`CeraEngine::apply_chat_template`].
  bool hasChatTemplate() => _unsupportedOnWeb('CeraEngine.hasChatTemplate');

  /// `true` when `id` is registered as a control or user-defined
  /// special token in the model's GGUF metadata
  /// (`tokenizer.ggml.token_type` types `3` / `4`). Useful for
  /// output filtering — e.g. dropping `<|im_end|>` from streamed
  /// tokens before rendering them to a UI — and for token-class
  /// classification in analysis tools.
  ///
  /// Out-of-range IDs (>= vocab size) and regular vocab tokens
  /// both return `false`. Companion to [`Self::special_token_id`]
  /// which goes the other direction (name → ID).
  bool isSpecialToken(int id) => _unsupportedOnWeb('CeraEngine.isSpecialToken');

  /// Short summary of the loaded model (architecture, vocab size,
  /// max context, etc.). Returns a `Clone` of the stored metadata.
  ModelMetadata metadata() => _unsupportedOnWeb('CeraEngine.metadata');

  /// Open a new [`Session`] sharing this engine's model + tokenizer
  /// by `Arc` clone. The returned session outlives `&self`; the
  /// engine keeps the shared state live for every session it hands
  /// out. Cheap — no model load, just config + state allocation.
  Session newSession(SessionConfig config) => _unsupportedOnWeb('CeraEngine.newSession');

  /// Look up a special token by name (e.g. `<|im_start|>`,
  /// `<|im_end|>`, `<|tool_call|>`). Returns `None` if the token
  /// isn't defined in the tokenizer's vocab.
  int? specialTokenId(String name) => _unsupportedOnWeb('CeraEngine.specialTokenId');

  /// The token id of `format`'s tool-call start marker (e.g.
  /// `<|tool_call_start|>`) in this model's vocab, for use as a lazy grammar
  /// trigger in `GenerateOpts.grammar_trigger_tokens`. `None` if the model's
  /// tokenizer lacks that special token.
  int? toolCallStartToken(ToolFormat format) => _unsupportedOnWeb('CeraEngine.toolCallStartToken');

  /// The tool-call format auto-detected from this model's architecture, or
  /// `None` if the architecture has no known tool convention.
  ToolFormat? toolFormat() => _unsupportedOnWeb('CeraEngine.toolFormat');

  /// Transcribe mono `f32` PCM audio (normalized to roughly `[-1.0, 1.0]`) to text using the
  /// model's trained `"Perform ASR."` chat mode. `sample_rate` must match the audio encoder's
  /// expected rate (resample beforehand if needed). Requires an audio-capable bundle; a text-only
  /// model returns an [`FfiError`] for unsupported modality.
  ///
  /// Blocking: runs a full prefill + greedy decode. Foreign async runtimes should wrap the call in
  /// `spawn_blocking` / its equivalent.
  String transcribe(List<double> pcm, int sampleRate) => _unsupportedOnWeb('CeraEngine.transcribe');

  /// Total vocabulary size — the number of distinct token IDs the
  /// model can emit. Sourced from the model's config (matches
  /// [`ModelMetadata::vocab_size`]) rather than the tokenizer's
  /// own count: in healthy models they match, but the model's
  /// config is the authoritative range for valid logit indices.
  int vocabSize() => _unsupportedOnWeb('CeraEngine.vocabSize');

  /// Wipe all KV prefix caches (both in-memory RAM tier and on-disk files).
  void wipeAllPrefixCaches() => _unsupportedOnWeb('CeraEngine.wipeAllPrefixCaches');
}

final class CeraEngineFfiCodec {
  static int lower(CeraEngine value) => _unsupportedOnWeb('CeraEngineFfiCodec.lower');
  static CeraEngine lift(int handle) => _unsupportedOnWeb('CeraEngineFfiCodec.lift');
}

/// Foreign-trait callback for download progress events from
/// [`BundleRepo::with_progress`]. Implementers (Kotlin class, Swift
/// class, Python subclass) drive a progress UI from these events.
///
/// All methods are required from foreign implementations (UniFFI
/// 0.31 foreign traits don't carry Rust default-impl fallbacks).
///
/// Threading: `on_progress` is invoked from the thread driving the
/// download. For sync `from_bundle_id` that's the caller's thread;
/// for `from_bundle_id_async` it's a tokio blocking worker. If your
/// progress UI requires marshalling onto a UI thread (`@MainActor`,
/// `runOnUiThread`, etc.), the implementer is responsible for the
/// dispatch.
abstract interface class DownloadProgressSink {
  /// Called periodically during a download. `bytes_downloaded` is
  /// monotonic across the same call's stream; `total_bytes` is the
  /// `Content-Length` reported by the server (may be `None` for
  /// chunked-transfer responses or when HEAD didn't surface a
  /// length). Same `url` value across all calls for one download
  /// — pattern-match on it to drive a per-file UI within a
  /// multi-file bundle download.
  ///
  /// Throttled by `cera-core` to ~256 KB granularity + one final
  /// callback at end-of-stream so the consumer always sees the
  /// final byte count.
  void onProgress(String url, int bytesDownloaded, int? totalBytes);
}

final class DownloadProgressSinkFfiCodec {
  static int lower(DownloadProgressSink value) => _unsupportedOnWeb('DownloadProgressSinkFfiCodec.lower');
  static DownloadProgressSink lift(int handle) => _unsupportedOnWeb('DownloadProgressSinkFfiCodec.lift');
}

/// Stateful Keyword Spotting detector executing pure-Rust forward inference.
final class FfiHotwordDetector {
  FfiHotwordDetector._();

  bool get isClosed => _unsupportedOnWeb('FfiHotwordDetector.isClosed');

  void close() => _unsupportedOnWeb('FfiHotwordDetector.close');

  /// Load a KWS model from in-memory GGUF bytes.
  static FfiHotwordDetector fromBytes(Uint8List bytes) => _unsupportedOnWeb('FfiHotwordDetector.fromBytes');

  /// Load a KWS model from a `.gguf` file path.
  static FfiHotwordDetector fromFile(String path) => _unsupportedOnWeb('FfiHotwordDetector.fromFile');

  /// Get default configuration suggested by model metadata.
  FfiHotwordConfig defaultConfig() => _unsupportedOnWeb('FfiHotwordDetector.defaultConfig');

  /// List of target keywords supported by this model.
  List<String> keywords() => _unsupportedOnWeb('FfiHotwordDetector.keywords');

  /// Process a full audio window and return probability scores for each keyword.
  List<double> processWindow(List<double> window) => _unsupportedOnWeb('FfiHotwordDetector.processWindow');
}

final class FfiHotwordDetectorFfiCodec {
  static int lower(FfiHotwordDetector value) => _unsupportedOnWeb('FfiHotwordDetectorFfiCodec.lower');
  static FfiHotwordDetector lift(int handle) => _unsupportedOnWeb('FfiHotwordDetectorFfiCodec.lift');
}

/// Streaming Keyword Spotting manager with VAD gating and debounce state.
final class FfiHotwordIterator {
  FfiHotwordIterator._();

  bool get isClosed => _unsupportedOnWeb('FfiHotwordIterator.isClosed');

  void close() => _unsupportedOnWeb('FfiHotwordIterator.close');

  /// Load and construct a streaming hotword iterator from file paths.
  static FfiHotwordIterator fromFiles(String modelPath, String? vadModelPath, FfiHotwordConfig? config) => _unsupportedOnWeb('FfiHotwordIterator.fromFiles');

  /// Process a streaming audio chunk and return a detection event if triggered.
  FfiHotwordEvent? processChunk(List<double> chunk) => _unsupportedOnWeb('FfiHotwordIterator.processChunk');

  /// Reset iterator state, ring buffer, and debounce timers.
  void reset() => _unsupportedOnWeb('FfiHotwordIterator.reset');
}

final class FfiHotwordIteratorFfiCodec {
  static int lower(FfiHotwordIterator value) => _unsupportedOnWeb('FfiHotwordIteratorFfiCodec.lower');
  static FfiHotwordIterator lift(int handle) => _unsupportedOnWeb('FfiHotwordIteratorFfiCodec.lift');
}

/// Stateful Silero Voice Activity Detector (VAD) session.
final class FfiSileroVad {
  FfiSileroVad._();

  bool get isClosed => _unsupportedOnWeb('FfiSileroVad.isClosed');

  void close() => _unsupportedOnWeb('FfiSileroVad.close');

  /// Load a Silero VAD model from in-memory GGUF bytes.
  static FfiSileroVad fromBytes(Uint8List bytes) => _unsupportedOnWeb('FfiSileroVad.fromBytes');

  /// Load a Silero VAD model from a `.gguf` file path.
  static FfiSileroVad fromFile(String path) => _unsupportedOnWeb('FfiSileroVad.fromFile');

  /// Process an entire audio buffer and return speech timestamps.
  List<FfiSpeechTimestamp> getSpeechTimestamps(List<double> audio, FfiVadSampleRate rate, FfiVadConfig? config) => _unsupportedOnWeb('FfiSileroVad.getSpeechTimestamps');

  /// Process a single chunk of audio and return the speech probability in `[0.0, 1.0]`.
  ///
  /// - 16 kHz: chunk must have exactly 512 samples.
  /// - 8 kHz: chunk must have exactly 256 samples.
  double processChunk(List<double> chunk, FfiVadSampleRate rate) => _unsupportedOnWeb('FfiSileroVad.processChunk');

  /// Reset recurrent state tensors and streaming context to zeros.
  void reset() => _unsupportedOnWeb('FfiSileroVad.reset');
}

final class FfiSileroVadFfiCodec {
  static int lower(FfiSileroVad value) => _unsupportedOnWeb('FfiSileroVadFfiCodec.lower');
  static FfiSileroVad lift(int handle) => _unsupportedOnWeb('FfiSileroVadFfiCodec.lift');
}

/// Stateful speech boundary detector for live audio streams.
final class FfiVadIterator {
  FfiVadIterator._();

  bool get isClosed => _unsupportedOnWeb('FfiVadIterator.isClosed');

  void close() => _unsupportedOnWeb('FfiVadIterator.close');

  /// Create a new streaming speech boundary iterator.
  static FfiVadIterator create(FfiVadSampleRate rate, FfiVadConfig? config) => _unsupportedOnWeb('FfiVadIterator.create');

  /// Flush any pending in-flight speech segment at the end of an audio stream.
  FfiVadEvent? flush() => _unsupportedOnWeb('FfiVadIterator.flush');

  /// Process a single chunk of audio and return any speech start or end event.
  FfiVadEvent? processChunk(FfiSileroVad vad, List<double> chunk) => _unsupportedOnWeb('FfiVadIterator.processChunk');

  /// Reset iterator state.
  void reset() => _unsupportedOnWeb('FfiVadIterator.reset');
}

final class FfiVadIteratorFfiCodec {
  static int lower(FfiVadIterator value) => _unsupportedOnWeb('FfiVadIteratorFfiCodec.lower');
  static FfiVadIterator lift(int handle) => _unsupportedOnWeb('FfiVadIteratorFfiCodec.lift');
}

/// A loaded LoRA adapter, ready to attach to a [`Session`] via
/// [`Session::attach_lora`]. Load it once and share the handle across sessions —
/// it's reference-counted internally, so attaching to multiple sessions doesn't
/// re-parse or re-allocate the factors.
final class LoraAdapters {
  LoraAdapters._();

  bool get isClosed => _unsupportedOnWeb('LoraAdapters.isClosed');

  void close() => _unsupportedOnWeb('LoraAdapters.close');

  /// Load a llama.cpp-format GGUF adapter (`convert_lora_to_gguf` output) from
  /// a local path. `alpha` is read from the adapter's `adapter.lora.alpha`
  /// metadata (missing ⇒ scale = 1).
  static LoraAdapters fromGguf(String path) => _unsupportedOnWeb('LoraAdapters.fromGguf');

  /// Load a PEFT `.safetensors` adapter from a local path. PEFT stores `alpha`
  /// in a sibling `adapter_config.json`, so pass it explicitly here (`None` ⇒
  /// scale = 1, i.e. `alpha == rank`).
  static LoraAdapters fromSafetensors(String path, double? alpha) => _unsupportedOnWeb('LoraAdapters.fromSafetensors');

  /// Number of `(layer, target)` low-rank deltas the adapter carries — for
  /// diagnostics / logging.
  int targetCount() => _unsupportedOnWeb('LoraAdapters.targetCount');
}

final class LoraAdaptersFfiCodec {
  static int lower(LoraAdapters value) => _unsupportedOnWeb('LoraAdaptersFfiCodec.lower');
  static LoraAdapters lift(int handle) => _unsupportedOnWeb('LoraAdaptersFfiCodec.lift');
}

/// Streaming sink for decode output. Foreign callers implement this
/// trait (Kotlin class, Swift class, Python subclass) and pass an
/// `Arc<dyn ModalitySink>` to [`Session::generate_streaming`] to
/// receive text chunks + thought chunks + audio frames + the finish reason
/// as they happen.
///
/// All methods are required from foreign implementations (UniFFI 0.31
/// foreign traits don't carry Rust's default-impl fallbacks). Callers
/// that don't care about a modality can provide an empty body.
///
/// Threading: every method is invoked on the same Rust thread running
/// `generate` (the decode thread). If the foreign runtime requires
/// marshalling onto a different thread (e.g. Swift's `@MainActor`) it
/// is the implementer's responsibility to dispatch the call there.
abstract interface class ModalitySink {
  /// Called with each chunk of reasoning or chain-of-thought text
  /// extracted from thinking delimiters (<think>...</think>).
  void onThoughtChunk(String text);
  /// Called with each chunk of generated user-facing text
  /// as soon as valid characters are produced.
  void onTextChunk(String text);
  /// Called with each chunk of generated PCM audio samples. Not
  /// called for text-only models; LFM2-Audio-class models emit here.
  /// The `sample_rate` is the model's native output rate (typically
  /// 24000 for LFM2-Audio) and is stable across the whole generate
  /// call.
  void onAudioFrames(List<double> pcm, int sampleRate);
  /// Called exactly once per [`Session::generate_streaming`] call,
  /// as the last thing before the wrapper returns. Fires for both
  /// success (`MaxTokens`, `Stop`, `Cancelled`, `ContextFull`) and
  /// failure paths: on error the wrapper synthesizes a
  /// [`FinishReason::Error`] so foreign consumers have a reliable
  /// end-of-stream signal regardless of how the call exits.
  void onDone(FinishReason reason);
}

final class ModalitySinkFfiCodec {
  static int lower(ModalitySink value) => _unsupportedOnWeb('ModalitySinkFfiCodec.lower');
  static ModalitySink lift(int handle) => _unsupportedOnWeb('ModalitySinkFfiCodec.lift');
}

/// Zero-dependency PII Classifier for named entity recognition.
final class PiiClassifier {
  PiiClassifier._();

  bool get isClosed => _unsupportedOnWeb('PiiClassifier.isClosed');

  void close() => _unsupportedOnWeb('PiiClassifier.close');

  /// Load a base model with a separate LoRA classifier adapter.
  static PiiClassifier fromBaseAndAdapter(String basePath, String adapterPath) => _unsupportedOnWeb('PiiClassifier.fromBaseAndAdapter');

  /// Load a PII classification model from a local GGUF path.
  static PiiClassifier fromPath(String path) => _unsupportedOnWeb('PiiClassifier.fromPath');

  /// Detect PII entities in the input text.
  List<FfiEntitySpan> detect(String text) => _unsupportedOnWeb('PiiClassifier.detect');
}

final class PiiClassifierFfiCodec {
  static int lower(PiiClassifier value) => _unsupportedOnWeb('PiiClassifierFfiCodec.lower');
  static PiiClassifier lift(int handle) => _unsupportedOnWeb('PiiClassifierFfiCodec.lift');
}

/// Stateful inference handle. Wraps [`cera::Session`] behind a
/// `Mutex` so UniFFI's `Arc<Session>` shape works with methods that
/// need `&mut self` on the inner session (prefill, generate, reset).
///
/// Call [`CeraEngine::new_session`] to open a session; the engine's
/// `Arc<Model>` and `Arc<BpeTokenizer>` are cloned into the new
/// session so it outlives the engine handle across FFI calls.
final class Session {
  Session._();

  bool get isClosed => _unsupportedOnWeb('Session.isClosed');

  void close() => _unsupportedOnWeb('Session.close');

  /// Append PCM audio samples (mono `f32`, normalized to roughly
  /// `[-1.0, 1.0]`) at `sample_rate` Hz. The audio is encoded via
  /// the bundle's mmproj (`AudioEncoderWeights`) and prefilled
  /// into the LLM as soft tokens — see
  /// [`cera::Session::append_audio`] for the underlying flow.
  ///
  /// `CeraEngine::new_session` auto-attaches the encoder when the
  /// loaded bundle's `inference_type == LlamaCppLfm2AudioV1` and
  /// has `multimodal_projector` set in the manifest, so FFI
  /// consumers don't need any separate "load encoder" call.
  /// Bundles where the manifest omits `multimodal_projector`
  /// silently end up with no encoder attached (no log). Bundles
  /// where the file is named but fails to open or parse log a
  /// `tracing::warn!` at `CeraEngine` construction. Both cases
  /// surface here as a "no audio encoder attached" `Backend`
  /// error.
  ///
  /// `sample_rate` must be 16000 — resampling is out of scope.
  /// Callers should resample externally before passing samples in.
  ///
  /// **Marshaling cost**: UniFFI maps `Vec<f32>` to `List<Float>`
  /// in Kotlin and `[Float]` in Swift. The Kotlin side boxes each
  /// `Float` to `java.lang.Float`, a ~4× memory overhead vs the
  /// underlying `f32` wire bytes — negligible for O(seconds ×
  /// sample-rate) chunks but worth knowing if you're streaming
  /// continuous audio in tight loops.
  ///
  /// Errors:
  /// - `EmptyInput` either when `samples` is empty (fast-fail,
  /// enforced here for parity with `append_text` /
  /// `append_tokens`) **or** when the audio is too short to
  /// produce any encoder frames (e.g. shorter than one
  /// center-padded STFT window).
  /// - `UnsupportedModality` if the loaded model's
  /// [`ModalityCapabilities::audio_in`] is `false`.
  /// - `Backend(...)` for sample-rate mismatch, encoder/LLM
  /// `hidden_size` mismatch, or missing encoder. The latter
  /// includes both "manifest didn't list a mmproj" (no warn
  /// logged) and "mmproj listed but failed to open/parse"
  /// (warn logged at `CeraEngine::from_path`).
  /// - `ContextOverflow` / `Cancelled` propagate from the
  /// underlying prefill.
  void appendAudio(List<double> samples, int sampleRate) => _unsupportedOnWeb('Session.appendAudio');

  /// Append an encoded image (PNG / JPEG bytes, auto-detected) to the
  /// context. The image is decoded, resized, normalized, and run
  /// through the bundle's vision mmproj (`VisionEncoderWeights`), then
  /// prefilled into the LLM as soft tokens — see
  /// [`cera::Session::append_image`] for the underlying flow.
  ///
  /// `CeraEngine::new_session` auto-attaches the vision encoder when
  /// the loaded bundle's `inference_type` is a VL type with
  /// `multimodal_projector` set in the manifest, so FFI consumers
  /// don't need a separate "load encoder" call. Bundles whose
  /// manifest omits the mmproj end up with no encoder attached (no
  /// log); bundles where it's named but fails to open/parse log a
  /// `tracing::warn!` at `CeraEngine` construction. Both surface here
  /// as a "no vision encoder attached" `Backend` error.
  ///
  /// `max_long_size` controls the per-call cap on the longest side of
  /// the *encoded* image, with three cases distinguished so the
  /// session default stays reachable through FFI:
  /// - `None` — defer to the session default set via
  /// [`Self::set_image_max_long_size`] (no cap if none was set).
  /// - `Some(0)` — explicitly force *no cap* for this call, ignoring
  /// the session default.
  /// - `Some(n)` (`n > 0`) — cap this call at `n`, overriding the
  /// session default.
  ///
  /// When a cap applies, the resize target is shrunk
  /// (aspect-preserving) so its longer side is at most `n` pixels,
  /// floored at one aligned patch block (so a very small `n` can still
  /// round up to that minimum) — a quality/cost knob (smaller = fewer
  /// image tokens, faster, less detail). It only shrinks (never
  /// upscales) and takes precedence over the model's
  /// minimum-resolution floor. The cap bounds the *encode*, not the
  /// *decode* (a huge source image is still decoded, bounded by
  /// internal limits).
  ///
  /// **Placement matters.** Prefer driving multimodal turns through
  /// the chat template; calling this at the wrong stream position
  /// (outside the model's image-marker envelope) leaves the LLM
  /// unable to interpret the embeddings as visual content. See
  /// [`cera::Session::append_image`] for the marker recipe.
  ///
  /// Errors (capability is checked before emptiness, matching core):
  /// - `UnsupportedModality` if the loaded model's
  /// [`ModalityCapabilities::image_in`] is `false`.
  /// - `EmptyInput` when `bytes` is empty (on a VL session).
  /// - `Backend(...)` for image decode failure, missing vision
  /// encoder, or encoder/LLM `projection_dim` ≠ `hidden_size`
  /// mismatch.
  /// - `ContextOverflow` / `Cancelled` propagate from the
  /// underlying prefill.
  void appendImage(Uint8List bytes, int? maxLongSize) => _unsupportedOnWeb('Session.appendImage');

  /// Append raw text to the context, running a prefill over just
  /// the new tokens. `EmptyInput` error if `text` is empty.
  void appendText(String text) => _unsupportedOnWeb('Session.appendText');

  /// Append pre-tokenized IDs. Useful when the caller has its own
  /// tokenizer + chat-template pipeline.
  void appendTokens(List<int> tokens) => _unsupportedOnWeb('Session.appendTokens');

  /// Attach a [`LoraAdapters`] to this session (generated as `attachLora` in
  /// Swift/Kotlin — this is the engine's equivalent of a `setLoraAdapters`
  /// call). It's applied to every subsequent forward pass — generation **and**
  /// hidden-states extraction — until removed or replaced (hot-swap), and is
  /// preserved across [`Self::reset`]. Only affects tokens processed after the
  /// call (doesn't retroactively re-adapt cached KV).
  ///
  /// Two distinct failures, worth catching separately: [`FfiError::LoraParse`]
  /// means the adapter's dimensions don't match the loaded model, so the
  /// adapter or the pairing is wrong; [`FfiError::LoraUnsupportedByBackend`]
  /// means it fits but this backend has no hook for something it adapts, so
  /// the same adapter works on another backend (today: a mixture-of-experts
  /// adapter needs the CPU backend).
  void attachLora(LoraAdapters adapters) => _unsupportedOnWeb('Session.attachLora');

  /// Signal in-flight `generate()` to exit with
  /// `FinishReason::Cancelled` at the next between-token check.
  /// Safe from any thread. No-op if no `generate()` is running.
  void cancel() => _unsupportedOnWeb('Session.cancel');

  /// Capabilities reported by the loaded model. Cheap — reads a
  /// cached copy, no lock.
  ModalityCapabilities capabilities() => _unsupportedOnWeb('Session.capabilities');

  /// Clear the cancel flag without dropping any session state.
  /// Use this after observing a cancellation signal — either
  /// [`FfiError::Cancelled`] from `append_text` / `append_tokens`
  /// / `append_audio` (mid-prefill cancellation surfaces
  /// typed), or `finish_reason = "Cancelled"` on the
  /// [`GenerateOutput`] returned from `generate` (cancellation
  /// during decode is reported as an `Ok` with that finish
  /// reason rather than an `Err`) — when you want to resume
  /// work on the same session without losing the accumulated
  /// KV cache.
  ///
  /// Compared to [`Self::reset`]:
  /// - `clear_cancel`: keeps KV state + position + sampler
  /// intact; only flips the cancel atomic back to `false`.
  /// Use for "interrupted but continuing" flows.
  /// - `reset`: drops KV cache + position + last logits +
  /// re-seeds sampler. Use for "clear conversation" flows.
  ///
  /// Atomic-backed; no mutex acquire, infallible, safe from
  /// any thread (mirrors the shape of [`Self::cancel`]).
  void clearCancel() => _unsupportedOnWeb('Session.clearCancel');

  /// Returns default `GenerateOpts` for this session, pre-populated with
  /// advisory sampling defaults from the bundle manifest (if any) or standard defaults.
  GenerateOpts defaultGenerateOpts() => _unsupportedOnWeb('Session.defaultGenerateOpts');

  /// Run autoregressive decode and return all emitted text, tokens, and
  /// summary. Synchronous: the call blocks until the decode loop exits
  /// (`max_tokens`, EOS, `cancel()`, or error).
  GenerateOutput generate(GenerateOpts opts) => _unsupportedOnWeb('Session.generate');

  /// Async variant of [`Session::generate`] — runs buffered decode
  /// (returning every emitted token + a summary) on a tokio blocking
  /// worker so the caller's async context isn't stalled by the
  /// synchronous decode loop.
  ///
  /// Cancellation: dropping the returned future (Kotlin coroutine
  /// scope exit, Swift `Task.cancel`, Python `asyncio.Task.cancel`)
  /// triggers both an abort of the queued `spawn_blocking` task (so
  /// a not-yet-started decode never runs) and a
  /// [`Session::cancel`] call (so an in-flight decode exits at its
  /// next between-token check with [`FinishReason::Cancelled`]).
  /// Either path releases the session mutex; subsequent calls see
  /// a clean session. You can also call [`Session::cancel`]
  /// directly from any thread to trigger the same in-flight exit
  /// without dropping the future. See `AsyncCancelGuard` for the
  /// full rationale.
  ///
  /// On error the wrapper performs the same poisoned-mutex handling
  /// as sync [`Session::generate`]. `JoinError` from a panic in the
  /// blocking closure surfaces as [`FfiError::Backend`] with a
  /// diagnostic prefix.
  Future<GenerateOutput> generateAsync(GenerateOpts opts) => _unsupportedOnWeb('Session.generateAsync');

  /// Run autoregressive decode, streaming every text chunk (and audio
  /// frame, for audio-capable models) to a foreign [`ModalitySink`]
  /// as soon as it is produced. Returns only a [`GenerateSummary`]:
  /// text chunks are delivered through `sink.on_text_chunk`, not a
  /// return value.
  ///
  /// Synchronous: the call blocks on the decode thread and each
  /// `sink` method runs on that same thread before decoding
  /// continues.
  ///
  /// **Callback reentrancy: deadlock hazard.** The session mutex is
  /// held for the entire call, and sink callbacks run while that
  /// lock is held. Calling back into methods that also take the
  /// mutex ([`Session::append_text`], [`Session::append_tokens`],
  /// [`Session::generate`], [`Session::generate_streaming`],
  /// [`Session::reset`]) from inside a sink method will deadlock.
  /// [`Session::cancel`] and [`Session::position`] are atomic-backed
  /// and safe to call from the sink or from any other thread.
  ///
  /// Cancellation: call [`Session::cancel`] from any thread (or from
  /// inside a sink callback on this thread) to terminate the loop at
  /// the next between-token check; `sink.on_done` fires with
  /// [`FinishReason::Cancelled`].
  ///
  /// End-of-stream guarantee: `sink.on_done` fires exactly once per
  /// call, even on error paths. If the underlying decode returns an
  /// error before reaching its own `on_done` call (e.g.,
  /// `EmptyInput` with no prefill logits), the wrapper synthesizes
  /// a terminal `on_done(FinishReason::Error { message })` so
  /// foreign consumers have a reliable end-of-stream signal
  /// regardless of how the call exits.
  GenerateSummary generateStreaming(GenerateOpts opts, ModalitySink sink) => _unsupportedOnWeb('Session.generateStreaming');

  /// Async variant of [`Session::generate_streaming`] — delivers
  /// tokens and audio frames to the foreign [`ModalitySink`] as the
  /// decode loop produces them, from within a blocking worker so
  /// the caller's async runtime stays responsive.
  ///
  /// Sink callbacks run on the blocking worker thread that's
  /// executing the decode — **not** on the caller's async thread.
  /// The reentrancy hazard documented on
  /// [`Session::generate_streaming`] still applies: sink callbacks
  /// that call back into `append_text` / `generate*` / `reset` from
  /// inside the session will deadlock on the session mutex.
  /// [`Session::cancel`] and [`Session::position`] remain atomic-
  /// backed and safe to invoke from any thread (including from
  /// inside a callback).
  ///
  /// Cancellation: dropping the returned future fires the same
  /// abort + [`Session::cancel`] pair as [`Session::generate_async`]
  /// (see `AsyncCancelGuard`). For an in-flight decode, the loop
  /// exits with [`FinishReason::Cancelled`] and the sink's `on_done`
  /// fires on the blocking worker before the task completes —
  /// foreign consumers get the terminal signal even though they've
  /// already stopped awaiting. For a queued-but-not-started decode,
  /// abort cancels the task without ever running the closure; no
  /// sink callbacks fire for that case (the decode never began).
  Future<GenerateSummary> generateStreamingAsync(GenerateOpts opts, ModalitySink sink) => _unsupportedOnWeb('Session.generateStreamingAsync');

  /// Whether a LoRA adapter is currently attached to this session.
  bool hasLora() => _unsupportedOnWeb('Session.hasLora');

  /// Model hidden dimension `D`. Reshape a raw `[T*D]` byte buffer from
  /// [`Self::hidden_states_for_tokens`] into `[T][D]` with this. Lock-free
  /// (cached at construction), so — like `position()` — it's safe to call from
  /// a `generate_streaming` sink callback.
  int hiddenSize() => _unsupportedOnWeb('Session.hiddenSize');

  /// Like [`Self::hidden_states_for_tokens`] but tokenizes `text` first
  /// (Swift `hiddenStatesForText(text:)`). Returns the same LE-f32 byte layout.
  Uint8List hiddenStatesForText(String text) => _unsupportedOnWeb('Session.hiddenStatesForText');

  /// Per-token last-layer hidden states (post-final-RMSNorm — the llama.cpp
  /// `--pooling none` / `llama_get_embeddings_ith` vector) for `tokens`,
  /// returned as **little-endian f32 bytes**: `n_tokens * hidden_size * 4`
  /// bytes, row-major, token `t` channel `c` at `(t*D + c) * 4`.
  ///
  /// Bytes (UniFFI `Data` in Swift, `ByteArray` in Kotlin) rather than
  /// `List<Float>` to avoid Kotlin's per-element boxing on the potentially
  /// large `[T*D]` payload. Swift decodes via `Data.withUnsafeBytes`; reflects
  /// the active LoRA once that lands. Side-effect-free: does not disturb the
  /// session's generation KV.
  ///
  /// Like `append_*` / `generate`, this holds the session mutex for the
  /// duration of the compute, so it must NOT be called re-entrantly from
  /// within a `generate_streaming` sink callback (would self-deadlock).
  ///
  /// Errors: `EmptyInput` on empty input; `UnsupportedModality` if the backend
  /// doesn't implement hidden-state extraction; `InvalidToken` if any id is
  /// `>= vocab_size`.
  Uint8List hiddenStatesForTokens(List<int> tokens) => _unsupportedOnWeb('Session.hiddenStatesForTokens');

  /// Mean-pooled hidden state — a single `[hidden_size]` vector (the common
  /// classifier path: pool in Rust, ship `D` floats not `T*D`). Returned as
  /// `[Float]` / `List<Float>`; only `D` elements, so boxing is negligible.
  List<double> hiddenStatesMeanPooled(List<int> tokens) => _unsupportedOnWeb('Session.hiddenStatesMeanPooled');

  /// Current KV position — how many tokens live in the cache.
  /// Atomic-backed; safe to call from a different thread while
  /// `generate()` is in flight.
  int position() => _unsupportedOnWeb('Session.position');

  /// Remove any attached LoRA adapter, returning to base-model inference.
  void removeLora() => _unsupportedOnWeb('Session.removeLora');

  /// Drop cached state + resample the seed. After `reset()` the
  /// session behaves like a freshly-opened one (same
  /// model/tokenizer/config, no accumulated context).
  ///
  /// Returns `Result` so a poisoned-mutex case surfaces as an error
  /// instead of panicking across the FFI boundary.
  void reset() => _unsupportedOnWeb('Session.reset');

  /// Append a multimodal message, automatically enforcing model-canonical
  /// media ordering, boundary token envelopes, and sample rate normalization.
  void sendMessage(UserMessage message) => _unsupportedOnWeb('Session.sendMessage');

  /// Append a multimodal message and run generation synchronously while holding
  /// the session lock continuously across prefill and decode.
  GenerateOutput sendMessageAndGenerate(UserMessage message, GenerateOpts opts) => _unsupportedOnWeb('Session.sendMessageAndGenerate');

  /// Append a multimodal message and run streaming generation while holding
  /// the session lock continuously across prefill and decode.
  GenerateSummary sendMessageStreaming(UserMessage message, GenerateOpts opts, ModalitySink sink) => _unsupportedOnWeb('Session.sendMessageStreaming');

  /// Set a session-default cap on the longest side of an appended
  /// image, in pixels (`None` = no cap). Unlike the per-call
  /// `max_long_size` argument to [`Self::append_image`], this default
  /// is honored by every image-append path the session drives —
  /// including chat-template flows — so a host can configure the
  /// image-encode budget once. See [`Self::append_image`] for the cap
  /// semantics (shrinks the encoded target, never upscales, takes
  /// precedence over the model's minimum-resolution floor).
  void setImageMaxLongSize(int? maxLongSize) => _unsupportedOnWeb('Session.setImageMaxLongSize');
}

final class SessionFfiCodec {
  static int lower(Session value) => _unsupportedOnWeb('SessionFfiCodec.lower');
  static Session lift(int handle) => _unsupportedOnWeb('SessionFfiCodec.lift');
}

/// Mirrors the native entry point. `dynamicLibrary` is `Object?` here
/// because its real type is `ffi.DynamicLibrary`, which does not exist on
/// this platform. Any call site that passes one cannot compile for web
/// regardless, so nothing is lost by widening it.
void configureDefaultBindings({Object? dynamicLibrary, String? libraryPath}) =>
    _unsupportedOnWeb('configureDefaultBindings');

/// No-op: there are no bindings to reset. Deliberately does not throw, so
/// that teardown in a `finally` cannot mask the original error.
void resetDefaultBindings() {}

/// Version string of the `cera-ffi` crate. Useful as a smoke test
/// from the foreign-language side — if this is callable, the binding
/// pipeline works end-to-end.
String ceraFfiVersion() => _unsupportedOnWeb('ceraFfiVersion');

/// One-line CPU backend report for this host — the resolved SIMD tier plus the
/// detected feature flags, e.g. `cpu: tier=neon+dotprod [neon dotprod]`. A host
/// property independent of any loaded model; callable without an engine. Handy
/// for telemetry and bug reports (tells you which kernel path actually ran).
String cpuBackendReport() => _unsupportedOnWeb('cpuBackendReport');

/// Detect the tool-call format for a model architecture string (e.g.
/// `"lfm2"`, `"qwen3"`). Returns `None` for architectures with no known
/// convention — the caller may still choose a format explicitly.
ToolFormat? detectToolFormat(String architecture) => _unsupportedOnWeb('detectToolFormat');

/// Default KWS configuration parameters.
FfiHotwordConfig hotwordDefaultConfig() => _unsupportedOnWeb('hotwordDefaultConfig');

/// List every bundle published on `LiquidAI/LeapBundles`, so a picker
/// can offer `<name>, <quant>` pairs instead of making the user type a
/// bundle id. Pair with [`CeraEngine::from_bundle_id`], which takes
/// exactly these two strings.
///
/// One blocking HTTP GET with a 30 s timeout and no retry. Prefer
/// [`list_leap_bundles_async`] anywhere a UI thread is involved: this
/// twin stalls the calling thread for the whole round-trip.
///
/// Needs no [`BundleRepo`]: the catalog is a single small JSON
/// response and is deliberately not cached, so a picker opened twice
/// in one session reflects newly published bundles.
List<LeapBundleEntry> listLeapBundles() => _unsupportedOnWeb('listLeapBundles');

/// Async variant of [`list_leap_bundles`]: moves the blocking HTTP
/// round-trip onto a tokio blocking worker so a coroutine, a Swift
/// `async` context or a Dart `Future` can await the catalog without
/// stalling the thread that asked for it.
///
/// `async_runtime = "tokio"` is load-bearing, not decoration: it is
/// what makes uniffi poll this future inside a tokio context. Without
/// it the foreign executor drives the future with no runtime
/// installed and the `spawn_blocking` below panics with "must be
/// called from the context of a Tokio 1.x runtime" on the very first
/// call.
///
/// Cancellation: dropping the returned future aborts the task if it
/// has not started, so a dismissed picker does not leave a 30 s
/// blocking GET queued on the pool. A request already in flight runs
/// to completion; `reqwest::blocking` offers nothing to interrupt, and
/// the response is small.
Future<List<LeapBundleEntry>> listLeapBundlesAsync() => _unsupportedOnWeb('listLeapBundlesAsync');

/// Parse tool calls out of generated model text for the given `format`.
/// Returns an empty list when the reply contains no tool call (the model
/// answered in prose). Errors only when a call section is present but
/// unrecoverably malformed.
List<ToolCall> parseToolCalls(String text, ToolFormat format) => _unsupportedOnWeb('parseToolCalls');

/// Default VAD configuration parameters.
FfiVadConfig sileroVadDefaultConfig() => _unsupportedOnWeb('sileroVadDefaultConfig');

/// Build a GBNF grammar string constraining output to a valid call for one
/// of `tools`, in `format`. Put the result in `GenerateOpts.grammar` and set
/// `GenerateOpts.grammar_trigger_tokens` (see
/// [`CeraEngine::tool_call_start_token`]) for a lazy tool-call trigger.
String toolGrammar(List<ToolDef> tools, ToolFormat format) => _unsupportedOnWeb('toolGrammar');
