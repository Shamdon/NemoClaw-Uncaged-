// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync, execSync } from "node:child_process";
import type { PluginLogger, NemoClawConfig } from "../index.js";
import {
  loadOnboardConfig,
  saveOnboardConfig,
  type EndpointType,
  type NemoClawOnboardConfig,
} from "../onboard/config.js";
import { promptInput, promptConfirm, promptSelect } from "../onboard/prompt.js";
import { validateApiKey, maskApiKey } from "../onboard/validate.js";

export interface OnboardOptions {
  apiKey?: string;
  endpoint?: string;
  ncpPartner?: string;
  endpointUrl?: string;
  model?: string;
  logger: PluginLogger;
  pluginConfig: NemoClawConfig;
}

const ENDPOINT_TYPES: EndpointType[] = [
  "build",
  "ncp",
  "openai",
  "anthropic",
  "groq",
  "together",
  "mistral",
  "google",
  "huggingface",
  "fireworks",
  "openrouter",
  "ollama",
  "lm-studio",
  "localai",
  "nim-local",
  "vllm",
  "custom",
];
const SUPPORTED_ENDPOINT_TYPES: EndpointType[] = [
  "build",
  "ncp",
  "openai",
  "anthropic",
  "groq",
  "together",
  "mistral",
  "google",
  "huggingface",
  "fireworks",
  "openrouter",
  "ollama",
  "lm-studio",
  "localai",
];

function isExperimentalEnabled(): boolean {
  return process.env.NEMOCLAW_EXPERIMENTAL === "1";
}

const BUILD_ENDPOINT_URL = "https://integrate.api.nvidia.com/v1";
const HOST_GATEWAY_URL = "http://host.openshell.internal";

// Endpoint URLs for managed cloud providers
const PROVIDER_ENDPOINT_URLS: Partial<Record<EndpointType, string>> = {
  build: BUILD_ENDPOINT_URL,
  openai: "https://api.openai.com/v1",
  anthropic: "https://api.anthropic.com/v1",
  groq: "https://api.groq.com/openai/v1",
  together: "https://api.together.xyz/v1",
  mistral: "https://api.mistral.ai/v1",
  google: "https://generativelanguage.googleapis.com/v1beta/openai",
  huggingface: "https://api-inference.huggingface.co/v1",
  fireworks: "https://api.fireworks.ai/inference/v1",
  openrouter: "https://openrouter.ai/api/v1",
};

// Per-provider API key env var names
const PROVIDER_CREDENTIAL_ENVS: Partial<Record<EndpointType, string>> = {
  build: "NVIDIA_API_KEY",
  ncp: "NVIDIA_API_KEY",
  custom: "NVIDIA_API_KEY",
  openai: "OPENAI_API_KEY",
  anthropic: "ANTHROPIC_API_KEY",
  groq: "GROQ_API_KEY",
  together: "TOGETHER_API_KEY",
  mistral: "MISTRAL_API_KEY",
  google: "GOOGLE_API_KEY",
  huggingface: "HF_TOKEN",
  fireworks: "FIREWORKS_API_KEY",
  openrouter: "OPENROUTER_API_KEY",
  "nim-local": "NIM_API_KEY",
  vllm: "OPENAI_API_KEY",
  ollama: "OPENAI_API_KEY",
  // lm-studio and localai require no real API key — placeholder only
  "lm-studio": "OPENAI_API_KEY",
  localai: "OPENAI_API_KEY",
};

// Provider-specific model catalogs
const PROVIDER_MODELS: Partial<Record<EndpointType, Array<{ id: string; label: string }>>> = {
  build: [
    { id: "nvidia/nemotron-3-super-120b-a12b", label: "Nemotron 3 Super 120B" },
    { id: "nvidia/llama-3.1-nemotron-ultra-253b-v1", label: "Nemotron Ultra 253B" },
    { id: "nvidia/llama-3.3-nemotron-super-49b-v1.5", label: "Nemotron Super 49B v1.5" },
    { id: "nvidia/nemotron-3-nano-30b-a3b", label: "Nemotron 3 Nano 30B" },
  ],
  openai: [
    { id: "gpt-4o", label: "GPT-4o" },
    { id: "gpt-4o-mini", label: "GPT-4o Mini" },
    { id: "o3-mini", label: "o3-mini" },
    { id: "o1", label: "o1" },
  ],
  anthropic: [
    { id: "claude-3-7-sonnet-20250219", label: "Claude 3.7 Sonnet" },
    { id: "claude-3-5-sonnet-20241022", label: "Claude 3.5 Sonnet" },
    { id: "claude-3-5-haiku-20241022", label: "Claude 3.5 Haiku" },
    { id: "claude-3-opus-20240229", label: "Claude 3 Opus" },
  ],
  groq: [
    { id: "llama-3.3-70b-versatile", label: "Llama 3.3 70B Versatile" },
    { id: "deepseek-r1-distill-llama-70b", label: "DeepSeek R1 Distill Llama 70B" },
    { id: "mixtral-8x7b-32768", label: "Mixtral 8x7B" },
    { id: "llama-3.1-8b-instant", label: "Llama 3.1 8B Instant" },
  ],
  together: [
    { id: "meta-llama/Llama-3.3-70B-Instruct-Turbo", label: "Llama 3.3 70B Instruct Turbo" },
    { id: "deepseek-ai/DeepSeek-R1", label: "DeepSeek R1" },
    { id: "mistralai/Mixtral-8x7B-Instruct-v0.1", label: "Mixtral 8x7B Instruct" },
    { id: "Qwen/QwQ-32B-Preview", label: "Qwen QwQ 32B Preview" },
  ],
  mistral: [
    { id: "mistral-large-latest", label: "Mistral Large" },
    { id: "mistral-small-latest", label: "Mistral Small" },
    { id: "codestral-latest", label: "Codestral" },
    { id: "open-mistral-nemo", label: "Mistral Nemo (open)" },
  ],
  google: [
    { id: "gemini-2.0-flash", label: "Gemini 2.0 Flash" },
    { id: "gemini-2.0-flash-thinking-exp", label: "Gemini 2.0 Flash Thinking (exp)" },
    { id: "gemini-1.5-pro", label: "Gemini 1.5 Pro" },
    { id: "gemini-1.5-flash", label: "Gemini 1.5 Flash" },
  ],
  // Hugging Face Inference API — popular open-source models
  huggingface: [
    { id: "meta-llama/Llama-3.3-70B-Instruct", label: "Llama 3.3 70B Instruct" },
    { id: "Qwen/Qwen2.5-72B-Instruct", label: "Qwen 2.5 72B Instruct" },
    { id: "mistralai/Mistral-7B-Instruct-v0.3", label: "Mistral 7B Instruct v0.3" },
    { id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", label: "DeepSeek R1 Distill Qwen 7B" },
  ],
  // Fireworks AI — fast open-source model hosting
  fireworks: [
    {
      id: "accounts/fireworks/models/llama-v3p3-70b-instruct",
      label: "Llama 3.3 70B Instruct",
    },
    { id: "accounts/fireworks/models/deepseek-r1", label: "DeepSeek R1" },
    {
      id: "accounts/fireworks/models/qwen2p5-72b-instruct",
      label: "Qwen 2.5 72B Instruct",
    },
    {
      id: "accounts/fireworks/models/mixtral-8x22b-instruct",
      label: "Mixtral 8x22B Instruct",
    },
  ],
  // OpenRouter — aggregates providers; models marked :free have no usage cost
  openrouter: [
    { id: "meta-llama/llama-3.3-70b-instruct:free", label: "Llama 3.3 70B Instruct (free)" },
    { id: "deepseek/deepseek-r1:free", label: "DeepSeek R1 (free)" },
    { id: "google/gemma-3-27b-it:free", label: "Gemma 3 27B Instruct (free)" },
    { id: "mistralai/mistral-7b-instruct:free", label: "Mistral 7B Instruct (free)" },
  ],
  // lm-studio and localai: no static catalog — live-discovered from running server
};

// Fallback NVIDIA model list (used for NCP, nim-local, vllm, custom, ollama)
const DEFAULT_MODELS = PROVIDER_MODELS.build!;

function resolveProfile(endpointType: EndpointType): string {
  switch (endpointType) {
    case "build":
      return "default";
    case "ncp":
    case "custom":
      return "ncp";
    case "openai":
      return "openai";
    case "anthropic":
      return "anthropic";
    case "groq":
      return "groq";
    case "together":
      return "together";
    case "mistral":
      return "mistral";
    case "google":
      return "google";
    case "huggingface":
      return "huggingface";
    case "fireworks":
      return "fireworks";
    case "openrouter":
      return "openrouter";
    case "ollama":
      return "ollama";
    case "lm-studio":
      return "lm-studio";
    case "localai":
      return "localai";
    case "nim-local":
      return "nim-local";
    case "vllm":
      return "vllm";
  }
}

function resolveProviderName(endpointType: EndpointType): string {
  switch (endpointType) {
    case "build":
      return "nvidia-nim";
    case "ncp":
    case "custom":
      return "nvidia-ncp";
    case "openai":
      return "openai";
    case "anthropic":
      return "anthropic";
    case "groq":
      return "groq";
    case "together":
      return "together";
    case "mistral":
      return "mistral";
    case "google":
      return "google";
    case "huggingface":
      return "huggingface";
    case "fireworks":
      return "fireworks";
    case "openrouter":
      return "openrouter";
    case "ollama":
      return "ollama-local";
    case "lm-studio":
      return "lm-studio-local";
    case "localai":
      return "localai-local";
    case "nim-local":
      return "nim-local";
    case "vllm":
      return "vllm-local";
  }
}

function resolveCredentialEnv(endpointType: EndpointType): string {
  return PROVIDER_CREDENTIAL_ENVS[endpointType] ?? "OPENAI_API_KEY";
}

function isNonInteractive(opts: OnboardOptions): boolean {
  if (!opts.endpoint || !opts.model) return false;
  const ep = opts.endpoint as EndpointType;
  if (endpointRequiresApiKey(ep) && !opts.apiKey) return false;
  // These endpoints require a custom URL to be specified
  const requiresCustomUrl = ep === "ncp" || ep === "nim-local" || ep === "custom";
  if (requiresCustomUrl && !opts.endpointUrl) return false;
  if (ep === "ncp" && !opts.ncpPartner) return false;
  return true;
}

// Local runtimes that serve an OpenAI-compatible API without any authentication
const KEYLESS_ENDPOINTS = new Set<EndpointType>(["vllm", "ollama", "lm-studio", "localai"]);

function endpointRequiresApiKey(endpointType: EndpointType): boolean {
  return !KEYLESS_ENDPOINTS.has(endpointType);
}

function defaultCredentialForEndpoint(endpointType: EndpointType): string {
  switch (endpointType) {
    case "vllm":
      return "dummy";
    case "ollama":
    case "lm-studio":
    case "localai":
      // These servers ignore the Authorization header entirely; use a
      // non-empty placeholder so OpenShell's openai provider is satisfied.
      return "no-key";
    default:
      return "";
  }
}

function getApiKeyDocsUrl(endpointType: EndpointType): string {
  switch (endpointType) {
    case "build":
      return "https://build.nvidia.com/settings/api-keys";
    case "ncp":
      return "https://www.nvidia.com/en-us/ai/cloud-partners/";
    case "openai":
      return "https://platform.openai.com/api-keys";
    case "anthropic":
      return "https://console.anthropic.com/settings/keys";
    case "groq":
      return "https://console.groq.com/keys";
    case "together":
      return "https://api.together.xyz/settings/api-keys";
    case "mistral":
      return "https://console.mistral.ai/api-keys/";
    case "google":
      return "https://aistudio.google.com/app/apikey";
    case "huggingface":
      return "https://huggingface.co/settings/tokens";
    case "fireworks":
      return "https://fireworks.ai/account/api-keys";
    case "openrouter":
      return "https://openrouter.ai/settings/keys";
    default:
      return "https://build.nvidia.com/settings/api-keys";
  }
}

interface LocalRuntimeStatus {
  ollama: { installed: boolean; running: boolean };
  lmStudio: { running: boolean };
  localai: { running: boolean };
}

function detectLocalRuntimes(): LocalRuntimeStatus {
  return {
    ollama: {
      installed: testCommand("command -v ollama >/dev/null 2>&1"),
      running: testCommand("curl -sf http://localhost:11434/api/tags >/dev/null 2>&1"),
    },
    lmStudio: {
      running: testCommand("curl -sf http://localhost:1234/v1/models >/dev/null 2>&1"),
    },
    localai: {
      running: testCommand("curl -sf http://localhost:8080/v1/models >/dev/null 2>&1"),
    },
  };
}

function testCommand(command: string): boolean {
  try {
    execSync(command, { encoding: "utf-8", stdio: "ignore", shell: "/bin/bash" });
    return true;
  } catch {
    return false;
  }
}

function showConfig(config: NemoClawOnboardConfig, logger: PluginLogger): void {
  logger.info(`  Endpoint:    ${config.endpointType} (${config.endpointUrl})`);
  if (config.ncpPartner) {
    logger.info(`  NCP Partner: ${config.ncpPartner}`);
  }
  logger.info(`  Model:       ${config.model}`);
  logger.info(`  Credential:  $${config.credentialEnv}`);
  logger.info(`  Profile:     ${config.profile}`);
  logger.info(`  Onboarded:   ${config.onboardedAt}`);
}

async function promptEndpoint(local: LocalRuntimeStatus): Promise<EndpointType> {
  const ollamaHint = local.ollama.running
    ? "running on localhost:11434"
    : local.ollama.installed
      ? "installed — start with: ollama serve"
      : "localhost:11434";

  const options = [
    {
      label: "NVIDIA Build (build.nvidia.com)",
      value: "build",
      hint: "recommended — zero infra, free credits",
    },
    {
      label: "NVIDIA Cloud Partner (NCP)",
      value: "ncp",
      hint: "dedicated capacity, SLA-backed",
    },
    {
      label: "OpenAI (api.openai.com)",
      value: "openai",
      hint: "GPT-4o, o3-mini, o1 and more",
    },
    {
      label: "Anthropic (api.anthropic.com)",
      value: "anthropic",
      hint: "Claude 3.7 Sonnet, Claude 3.5 Sonnet and more",
    },
    {
      label: "Groq (api.groq.com)",
      value: "groq",
      hint: "ultra-fast inference — Llama 3.3, DeepSeek R1, Mixtral",
    },
    {
      label: "Together AI (api.together.xyz)",
      value: "together",
      hint: "open-source model hosting — Llama, DeepSeek, Qwen",
    },
    {
      label: "Mistral AI (api.mistral.ai)",
      value: "mistral",
      hint: "Mistral Large, Codestral, Mistral Nemo",
    },
    {
      label: "Google Gemini (generativelanguage.googleapis.com)",
      value: "google",
      hint: "Gemini 2.0 Flash, Gemini 1.5 Pro and more",
    },
    {
      label: "Hugging Face Inference API (api-inference.huggingface.co)",
      value: "huggingface",
      hint: "open-source models — Llama, Qwen, Mistral, DeepSeek",
    },
    {
      label: "Fireworks AI (api.fireworks.ai)",
      value: "fireworks",
      hint: "fast open-source hosting — Llama, DeepSeek, Mixtral, Qwen",
    },
    {
      label: "OpenRouter (openrouter.ai) — includes free models",
      value: "openrouter",
      hint: "aggregates 200+ models; :free models require no credits",
    },
    {
      label: `Ollama (local, no API key)`,
      value: "ollama",
      hint: ollamaHint,
    },
    {
      label: `LM Studio (local, no API key)`,
      value: "lm-studio",
      hint: local.lmStudio.running
        ? "running on localhost:1234"
        : "start LM Studio → Local Server tab",
    },
    {
      label: "LocalAI (local, no API key)",
      value: "localai",
      hint: local.localai.running
        ? "running on localhost:8080"
        : "self-hosted OpenAI-compatible runtime",
    },
  ];

  if (isExperimentalEnabled()) {
    options.push(
      {
        label: "Self-hosted NIM [experimental]",
        value: "nim-local",
        hint: "experimental — your own NIM container deployment",
      },
      {
        label: "Local vLLM [experimental]",
        value: "vllm",
        hint: "experimental — local vLLM server",
      },
    );
  }

  return (await promptSelect("Select your inference endpoint:", options)) as EndpointType;
}

function execOpenShell(args: string[]): string {
  return execFileSync("openshell", args, {
    encoding: "utf-8",
    stdio: ["pipe", "pipe", "pipe"],
  });
}

export async function cliOnboard(opts: OnboardOptions): Promise<void> {
  const { logger } = opts;
  const nonInteractive = isNonInteractive(opts);

  logger.info("NemoClaw Onboarding");
  logger.info("-------------------");

  // Step 0: Check existing config
  const existing = loadOnboardConfig();
  if (existing) {
    logger.info("");
    logger.info("Existing configuration found:");
    showConfig(existing, logger);
    logger.info("");

    if (!nonInteractive) {
      const reconfigure = await promptConfirm("Reconfigure?", false);
      if (!reconfigure) {
        logger.info("Keeping existing configuration.");
        return;
      }
    }
  }

  // Step 1: Endpoint Selection
  let endpointType: EndpointType;
  if (opts.endpoint) {
    if (!ENDPOINT_TYPES.includes(opts.endpoint as EndpointType)) {
      logger.error(
        `Invalid endpoint type: ${opts.endpoint}. Must be one of: ${ENDPOINT_TYPES.join(", ")}`,
      );
      return;
    }
    const ep = opts.endpoint as EndpointType;
    if (!SUPPORTED_ENDPOINT_TYPES.includes(ep)) {
      logger.warn(
        `Note: '${ep}' is experimental and may not work reliably.`,
      );
    }
    endpointType = ep;
  } else {
    const local = detectLocalRuntimes();
    // Auto-select a running local runtime only when experimental mode is on
    // (avoids surprising users who run these services for unrelated reasons).
    if (isExperimentalEnabled()) {
      if (local.lmStudio.running) {
        logger.info("Detected LM Studio on localhost:1234. Using it for onboarding.");
        endpointType = "lm-studio";
      } else if (local.localai.running) {
        logger.info("Detected LocalAI on localhost:8080. Using it for onboarding.");
        endpointType = "localai";
      } else if (local.ollama.running) {
        logger.info("Detected Ollama on localhost:11434. Using it for onboarding.");
        endpointType = "ollama";
      } else {
        endpointType = await promptEndpoint(local);
      }
    } else {
      endpointType = await promptEndpoint(local);
    }
  }

  // Step 2: Endpoint URL resolution
  let endpointUrl: string;
  let ncpPartner: string | null = null;

  // Check if this endpoint has a fixed well-known URL
  const fixedUrl = PROVIDER_ENDPOINT_URLS[endpointType];

  switch (endpointType) {
    case "build":
    case "openai":
    case "anthropic":
    case "groq":
    case "together":
    case "mistral":
    case "google":
    case "huggingface":
    case "fireworks":
    case "openrouter":
      // All managed cloud providers have a fixed, well-known endpoint
      endpointUrl = fixedUrl!;
      break;
    case "ncp":
      ncpPartner = opts.ncpPartner ?? (await promptInput("NCP partner name"));
      endpointUrl =
        opts.endpointUrl ??
        (await promptInput("NCP endpoint URL (e.g., https://partner.api.nvidia.com/v1)"));
      break;
    case "nim-local":
      endpointUrl =
        opts.endpointUrl ??
        (await promptInput("NIM endpoint URL", "http://nim-service.local:8000/v1"));
      break;
    case "vllm":
      endpointUrl = `${HOST_GATEWAY_URL}:8000/v1`;
      break;
    case "ollama":
      endpointUrl = opts.endpointUrl ?? `${HOST_GATEWAY_URL}:11434/v1`;
      break;
    case "lm-studio":
      endpointUrl = opts.endpointUrl ?? `${HOST_GATEWAY_URL}:1234/v1`;
      break;
    case "localai":
      endpointUrl = opts.endpointUrl ?? `${HOST_GATEWAY_URL}:8080/v1`;
      break;
    case "custom":
      endpointUrl = opts.endpointUrl ?? (await promptInput("Custom endpoint URL"));
      break;
  }

  if (!endpointUrl) {
    logger.error("No endpoint URL provided. Aborting.");
    return;
  }

  const credentialEnv = resolveCredentialEnv(endpointType);
  const requiresApiKey = endpointRequiresApiKey(endpointType);

  // Step 3: Credential
  let apiKey = defaultCredentialForEndpoint(endpointType);
  if (requiresApiKey) {
    if (opts.apiKey) {
      apiKey = opts.apiKey;
    } else {
      // Check the provider-specific env var; only fall back to NVIDIA_API_KEY for NVIDIA endpoints
      const isNvidiaEndpoint = endpointType === "build" || endpointType === "ncp";
      const envKey =
        process.env[credentialEnv] ?? (isNvidiaEndpoint ? process.env.NVIDIA_API_KEY : undefined);
      const envVarName = process.env[credentialEnv] ? credentialEnv : "NVIDIA_API_KEY";
      const apiKeyDocs = getApiKeyDocsUrl(endpointType);
      if (envKey) {
        logger.info(`Detected ${envVarName} in environment (${maskApiKey(envKey)})`);
        const useEnv = nonInteractive ? true : await promptConfirm("Use this key?");
        apiKey = useEnv ? envKey : await promptInput(`Enter your ${credentialEnv}`);
      } else {
        logger.info(`Get an API key from: ${apiKeyDocs}`);
        apiKey = await promptInput(`Enter your ${credentialEnv}`);
      }
    }
  } else {
    logger.info(
      `No API key required for ${endpointType}. Using local credential value '${apiKey}'.`,
    );
  }

  if (!apiKey) {
    logger.error("No API key provided. Aborting.");
    return;
  }

  // Step 4: Validate API Key
  // For local endpoints, validation is best-effort since the service may not be
  // running yet during onboarding.
  const isLocalEndpoint = KEYLESS_ENDPOINTS.has(endpointType) || endpointType === "nim-local";
  logger.info("");
  logger.info(`Validating ${requiresApiKey ? "credential" : "endpoint"} against ${endpointUrl}...`);
  const validation = await validateApiKey(apiKey, endpointUrl);

  if (!validation.valid) {
    if (isLocalEndpoint) {
      logger.warn(
        `Could not reach ${endpointUrl} (${validation.error ?? "unknown error"}). Continuing anyway — the service may not be running yet.`,
      );
    } else {
      logger.error(`API key validation failed: ${validation.error ?? "unknown error"}`);
      logger.info(`Check your key at: ${getApiKeyDocsUrl(endpointType)}`);
      return;
    }
  } else {
    logger.info(
      `${requiresApiKey ? "Credential" : "Endpoint"} valid. ${String(validation.models.length)} model(s) available.`,
    );
  }

  // Step 5: Model Selection
  let model: string;
  if (opts.model) {
    model = opts.model;
  } else {
    // Use provider-specific catalog, or fall back to live-discovered models
    const catalogModels = PROVIDER_MODELS[endpointType];
    let modelOptions: Array<{ label: string; value: string }>;

    if (validation.models.length > 0) {
      // Prefer live-discovered models so we always reflect what's actually available
      if (catalogModels) {
        // Cross-reference with catalog to surface friendly labels
        const catalogById = new Map(catalogModels.map((m) => [m.id, m.label]));
        modelOptions = validation.models.map((id) => ({
          label: catalogById.has(id) ? `${catalogById.get(id)!} (${id})` : id,
          value: id,
        }));
      } else {
        modelOptions = validation.models.map((id) => ({ label: id, value: id }));
      }
    } else if (catalogModels) {
      // No live response — fall back to static catalog
      modelOptions = catalogModels.map((m) => ({ label: `${m.label} (${m.id})`, value: m.id }));
    } else if (endpointType === "lm-studio" || endpointType === "localai") {
      // Local runtime with no loaded models yet — ask the user to type the model name
      logger.warn(
        `No models found at ${endpointUrl}. Make sure a model is loaded in ${endpointType === "lm-studio" ? "LM Studio (Local Server tab)" : "LocalAI"}, then re-run onboarding.`,
      );
      model = await promptInput("Enter model identifier (e.g. lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF)");
    } else {
      // Last-resort fallback to NVIDIA defaults
      modelOptions = DEFAULT_MODELS.map((m) => ({ label: `${m.label} (${m.id})`, value: m.id }));
    }

    // modelOptions may be undefined if we already set `model` above via promptInput
    if (!model) {
      model = await promptSelect("Select your primary model:", modelOptions!);
    }
  }

  // Step 6: Resolve profile
  const profile = resolveProfile(endpointType);
  const providerName = resolveProviderName(endpointType);

  // Step 7: Confirmation
  logger.info("");
  logger.info("Configuration summary:");
  logger.info(`  Endpoint:    ${endpointType} (${endpointUrl})`);
  if (ncpPartner) {
    logger.info(`  NCP Partner: ${ncpPartner}`);
  }
  logger.info(`  Model:       ${model}`);
  logger.info(
    `  API Key:     ${requiresApiKey ? maskApiKey(apiKey) : "not required (local provider)"}`,
  );
  logger.info(`  Credential:  $${credentialEnv}`);
  logger.info(`  Profile:     ${profile}`);
  logger.info(`  Provider:    ${providerName}`);
  logger.info("");

  if (!nonInteractive) {
    const proceed = await promptConfirm("Apply this configuration?");
    if (!proceed) {
      logger.info("Onboarding cancelled.");
      return;
    }
  }

  // Step 8: Apply
  logger.info("");
  logger.info("Applying configuration...");

  // 7a: Create/update provider
  // Anthropic uses its own provider type; all other providers are OpenAI-compatible
  const openShellProviderType = endpointType === "anthropic" ? "anthropic" : "openai";
  const openShellBaseUrlKey =
    endpointType === "anthropic" ? "ANTHROPIC_BASE_URL" : "OPENAI_BASE_URL";
  try {
    execOpenShell([
      "provider",
      "create",
      "--name",
      providerName,
      "--type",
      openShellProviderType,
      "--credential",
      `${credentialEnv}=${apiKey}`,
      "--config",
      `${openShellBaseUrlKey}=${endpointUrl}`,
    ]);
    logger.info(`Created provider: ${providerName}`);
  } catch (err) {
    const stderr =
      err instanceof Error && "stderr" in err ? String((err as { stderr: unknown }).stderr) : "";
    if (stderr.includes("AlreadyExists") || stderr.includes("already exists")) {
      try {
        execOpenShell([
          "provider",
          "update",
          providerName,
          "--credential",
          `${credentialEnv}=${apiKey}`,
          "--config",
          `${openShellBaseUrlKey}=${endpointUrl}`,
        ]);
        logger.info(`Updated provider: ${providerName}`);
      } catch (updateErr) {
        const updateStderr =
          updateErr instanceof Error && "stderr" in updateErr
            ? String((updateErr as { stderr: unknown }).stderr)
            : "";
        logger.error(`Failed to update provider: ${updateStderr || String(updateErr)}`);
        return;
      }
    } else {
      logger.error(`Failed to create provider: ${stderr || String(err)}`);
      return;
    }
  }

  // 7b: Set inference route
  try {
    execOpenShell(["inference", "set", "--provider", providerName, "--model", model]);
    logger.info(`Inference route set: ${providerName} -> ${model}`);
  } catch (err) {
    const stderr =
      err instanceof Error && "stderr" in err ? String((err as { stderr: unknown }).stderr) : "";
    logger.error(`Failed to set inference route: ${stderr || String(err)}`);
    return;
  }

  // 7c: Save config
  saveOnboardConfig({
    endpointType,
    endpointUrl,
    ncpPartner,
    model,
    profile,
    credentialEnv,
    onboardedAt: new Date().toISOString(),
  });

  // Step 9: Success
  logger.info("");
  logger.info("Onboarding complete!");
  logger.info("");
  logger.info(`  Endpoint:   ${endpointUrl}`);
  logger.info(`  Model:      ${model}`);
  logger.info(`  Credential: $${credentialEnv}`);
  logger.info("");
  logger.info("Next steps:");
  logger.info("  openclaw nemoclaw launch     # Bootstrap sandbox");
  logger.info("  openclaw nemoclaw status     # Check configuration");
}
