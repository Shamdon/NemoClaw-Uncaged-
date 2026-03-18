// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NemoClaw — OpenClaw Plugin for OpenShell
 *
 * Uses the real OpenClaw plugin API. Types defined locally are minimal stubs
 * that match the OpenClaw SDK interfaces available at runtime via
 * `openclaw/plugin-sdk`. We define them here because the SDK package is only
 * available inside the OpenClaw host process and cannot be imported at build
 * time.
 */

import type { Command } from "commander";
import { registerCliCommands } from "./cli.js";
import { handleSlashCommand } from "./commands/slash.js";
import { loadOnboardConfig } from "./onboard/config.js";

// ---------------------------------------------------------------------------
// OpenClaw Plugin SDK compatible types (mirrors openclaw/plugin-sdk)
// ---------------------------------------------------------------------------

/** Subset of OpenClawConfig that we actually read. */
export interface OpenClawConfig {
  [key: string]: unknown;
}

/** Logger provided by the plugin host. */
export interface PluginLogger {
  info(message: string): void;
  warn(message: string): void;
  error(message: string): void;
  debug(message: string): void;
}

/** Context passed to slash-command handlers. */
export interface PluginCommandContext {
  senderId?: string;
  channel: string;
  isAuthorizedSender: boolean;
  args?: string;
  commandBody: string;
  config: OpenClawConfig;
  from?: string;
  to?: string;
  accountId?: string;
}

/** Return value from a slash-command handler. */
export interface PluginCommandResult {
  text?: string;
  mediaUrl?: string;
  mediaUrls?: string[];
}

/** Registration shape for a slash command. */
export interface PluginCommandDefinition {
  name: string;
  description: string;
  acceptsArgs?: boolean;
  requireAuth?: boolean;
  handler: (ctx: PluginCommandContext) => PluginCommandResult | Promise<PluginCommandResult>;
}

/** Context passed to the CLI registrar callback. */
export interface PluginCliContext {
  program: Command;
  config: OpenClawConfig;
  workspaceDir?: string;
  logger: PluginLogger;
}

/** CLI registrar callback type. */
export type PluginCliRegistrar = (ctx: PluginCliContext) => void | Promise<void>;

/** Auth method for a provider plugin. */
export interface ProviderAuthMethod {
  type: string;
  envVar?: string;
  headerName?: string;
  label?: string;
}

/** Model entry in a provider's model catalog. */
export interface ModelProviderEntry {
  id: string;
  label: string;
  contextWindow?: number;
  maxOutput?: number;
}

/** Model catalog shape. */
export interface ModelProviderConfig {
  chat?: ModelProviderEntry[];
  completion?: ModelProviderEntry[];
}

/** Registration shape for a custom model provider. */
export interface ProviderPlugin {
  id: string;
  label: string;
  docsPath?: string;
  aliases?: string[];
  envVars?: string[];
  models?: ModelProviderConfig;
  auth: ProviderAuthMethod[];
}

/** Background service registration. */
export interface PluginService {
  id: string;
  start: (ctx: { config: OpenClawConfig; logger: PluginLogger }) => void | Promise<void>;
  stop?: (ctx: { config: OpenClawConfig; logger: PluginLogger }) => void | Promise<void>;
}

/**
 * The API object injected into the plugin's register function by the OpenClaw
 * host. Only the methods we actually call are listed here.
 */
export interface OpenClawPluginApi {
  id: string;
  name: string;
  version?: string;
  config: OpenClawConfig;
  pluginConfig?: Record<string, unknown>;
  logger: PluginLogger;
  registerCommand: (command: PluginCommandDefinition) => void;
  registerCli: (registrar: PluginCliRegistrar, opts?: { commands?: string[] }) => void;
  registerProvider: (provider: ProviderPlugin) => void;
  registerService: (service: PluginService) => void;
  resolvePath: (input: string) => string;
  on: (hookName: string, handler: (...args: unknown[]) => void) => void;
}

// ---------------------------------------------------------------------------
// Plugin-specific config (read from pluginConfig in openclaw.plugin.json)
// ---------------------------------------------------------------------------

export interface NemoClawConfig {
  blueprintVersion: string;
  blueprintRegistry: string;
  sandboxName: string;
  inferenceProvider: string;
}

const DEFAULT_PLUGIN_CONFIG: NemoClawConfig = {
  blueprintVersion: "latest",
  blueprintRegistry: "ghcr.io/nvidia/nemoclaw-blueprint",
  sandboxName: "openclaw",
  inferenceProvider: "nvidia",
};

export function getPluginConfig(api: OpenClawPluginApi): NemoClawConfig {
  const raw = api.pluginConfig ?? {};
  return {
    blueprintVersion:
      typeof raw["blueprintVersion"] === "string"
        ? raw["blueprintVersion"]
        : DEFAULT_PLUGIN_CONFIG.blueprintVersion,
    blueprintRegistry:
      typeof raw["blueprintRegistry"] === "string"
        ? raw["blueprintRegistry"]
        : DEFAULT_PLUGIN_CONFIG.blueprintRegistry,
    sandboxName:
      typeof raw["sandboxName"] === "string"
        ? raw["sandboxName"]
        : DEFAULT_PLUGIN_CONFIG.sandboxName,
    inferenceProvider:
      typeof raw["inferenceProvider"] === "string"
        ? raw["inferenceProvider"]
        : DEFAULT_PLUGIN_CONFIG.inferenceProvider,
  };
}

// ---------------------------------------------------------------------------
// Plugin entry point
// ---------------------------------------------------------------------------

export default function register(api: OpenClawPluginApi): void {
  // 1. Register /nemoclaw slash command (chat interface)
  api.registerCommand({
    name: "nemoclaw",
    description: "NemoClaw sandbox management (status, eject).",
    acceptsArgs: true,
    handler: (ctx) => handleSlashCommand(ctx, api),
  });

  // 2. Register `openclaw nemoclaw` CLI subcommands (commander.js)
  api.registerCli(
    (cliCtx) => {
      registerCliCommands(cliCtx, api);
    },
    { commands: ["nemoclaw"] },
  );

  // 3. Register all LLM providers — use onboard config if available
  const onboardCfg = loadOnboardConfig();
  const providerCredentialEnv = onboardCfg?.credentialEnv ?? "NVIDIA_API_KEY";
  const providerLabel = onboardCfg
    ? `NVIDIA NIM (${onboardCfg.endpointType}${onboardCfg.ncpPartner ? ` - ${onboardCfg.ncpPartner}` : ""})`
    : "NVIDIA NIM (build.nvidia.com)";

  // NVIDIA NIM — always registered as the primary NVIDIA provider
  api.registerProvider({
    id: "nvidia-nim",
    label: providerLabel,
    docsPath: "https://build.nvidia.com/docs",
    aliases: ["nvidia", "nim"],
    envVars: [providerCredentialEnv],
    models: {
      chat: [
        {
          id: "nvidia/nemotron-3-super-120b-a12b",
          label: "Nemotron 3 Super 120B (March 2026)",
          contextWindow: 131072,
          maxOutput: 8192,
        },
        {
          id: "nvidia/llama-3.1-nemotron-ultra-253b-v1",
          label: "Nemotron Ultra 253B",
          contextWindow: 131072,
          maxOutput: 4096,
        },
        {
          id: "nvidia/llama-3.3-nemotron-super-49b-v1.5",
          label: "Nemotron Super 49B v1.5",
          contextWindow: 131072,
          maxOutput: 4096,
        },
        {
          id: "nvidia/nemotron-3-nano-30b-a3b",
          label: "Nemotron 3 Nano 30B",
          contextWindow: 131072,
          maxOutput: 4096,
        },
      ],
    },
    auth: [
      {
        type: "bearer",
        envVar: providerCredentialEnv,
        headerName: "Authorization",
        label: `NVIDIA API Key (${providerCredentialEnv})`,
      },
    ],
  });

  // OpenAI — GPT-4o, o3-mini, o1
  api.registerProvider({
    id: "openai",
    label: "OpenAI",
    docsPath: "https://platform.openai.com/docs",
    aliases: ["gpt", "chatgpt"],
    envVars: ["OPENAI_API_KEY"],
    models: {
      chat: [
        { id: "gpt-4o", label: "GPT-4o", contextWindow: 128000, maxOutput: 16384 },
        { id: "gpt-4o-mini", label: "GPT-4o Mini", contextWindow: 128000, maxOutput: 16384 },
        { id: "o3-mini", label: "o3-mini", contextWindow: 200000, maxOutput: 100000 },
        { id: "o1", label: "o1", contextWindow: 200000, maxOutput: 100000 },
      ],
    },
    auth: [
      {
        type: "bearer",
        envVar: "OPENAI_API_KEY",
        headerName: "Authorization",
        label: "OpenAI API Key",
      },
    ],
  });

  // Anthropic — Claude 3.7, Claude 3.5
  api.registerProvider({
    id: "anthropic",
    label: "Anthropic",
    docsPath: "https://docs.anthropic.com",
    aliases: ["claude"],
    envVars: ["ANTHROPIC_API_KEY"],
    models: {
      chat: [
        {
          id: "claude-3-7-sonnet-20250219",
          label: "Claude 3.7 Sonnet",
          contextWindow: 200000,
          maxOutput: 8192,
        },
        {
          id: "claude-3-5-sonnet-20241022",
          label: "Claude 3.5 Sonnet",
          contextWindow: 200000,
          maxOutput: 8192,
        },
        {
          id: "claude-3-5-haiku-20241022",
          label: "Claude 3.5 Haiku",
          contextWindow: 200000,
          maxOutput: 8192,
        },
        {
          id: "claude-3-opus-20240229",
          label: "Claude 3 Opus",
          contextWindow: 200000,
          maxOutput: 4096,
        },
      ],
    },
    auth: [
      {
        type: "api-key",
        envVar: "ANTHROPIC_API_KEY",
        headerName: "x-api-key",
        label: "Anthropic API Key",
      },
    ],
  });

  // Groq — ultra-fast inference
  api.registerProvider({
    id: "groq",
    label: "Groq",
    docsPath: "https://console.groq.com/docs",
    aliases: ["groq-cloud"],
    envVars: ["GROQ_API_KEY"],
    models: {
      chat: [
        {
          id: "llama-3.3-70b-versatile",
          label: "Llama 3.3 70B Versatile",
          contextWindow: 128000,
          maxOutput: 32768,
        },
        {
          id: "deepseek-r1-distill-llama-70b",
          label: "DeepSeek R1 Distill Llama 70B",
          contextWindow: 128000,
          maxOutput: 16384,
        },
        {
          id: "mixtral-8x7b-32768",
          label: "Mixtral 8x7B",
          contextWindow: 32768,
          maxOutput: 32768,
        },
        {
          id: "llama-3.1-8b-instant",
          label: "Llama 3.1 8B Instant",
          contextWindow: 128000,
          maxOutput: 8000,
        },
      ],
    },
    auth: [
      {
        type: "bearer",
        envVar: "GROQ_API_KEY",
        headerName: "Authorization",
        label: "Groq API Key",
      },
    ],
  });

  // Together AI — open-source model hosting
  api.registerProvider({
    id: "together",
    label: "Together AI",
    docsPath: "https://docs.together.ai",
    aliases: ["together-ai", "togetherai"],
    envVars: ["TOGETHER_API_KEY"],
    models: {
      chat: [
        {
          id: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
          label: "Llama 3.3 70B Instruct Turbo",
          contextWindow: 131072,
          maxOutput: 8192,
        },
        {
          id: "deepseek-ai/DeepSeek-R1",
          label: "DeepSeek R1",
          contextWindow: 163840,
          maxOutput: 32768,
        },
        {
          id: "mistralai/Mixtral-8x7B-Instruct-v0.1",
          label: "Mixtral 8x7B Instruct",
          contextWindow: 32768,
          maxOutput: 32768,
        },
        {
          id: "Qwen/QwQ-32B-Preview",
          label: "Qwen QwQ 32B Preview",
          contextWindow: 32768,
          maxOutput: 16384,
        },
      ],
    },
    auth: [
      {
        type: "bearer",
        envVar: "TOGETHER_API_KEY",
        headerName: "Authorization",
        label: "Together AI API Key",
      },
    ],
  });

  // Mistral AI
  api.registerProvider({
    id: "mistral",
    label: "Mistral AI",
    docsPath: "https://docs.mistral.ai",
    aliases: ["mistral-ai"],
    envVars: ["MISTRAL_API_KEY"],
    models: {
      chat: [
        {
          id: "mistral-large-latest",
          label: "Mistral Large",
          contextWindow: 131072,
          maxOutput: 4096,
        },
        {
          id: "mistral-small-latest",
          label: "Mistral Small",
          contextWindow: 131072,
          maxOutput: 4096,
        },
        {
          id: "codestral-latest",
          label: "Codestral",
          contextWindow: 262144,
          maxOutput: 8192,
        },
        {
          id: "open-mistral-nemo",
          label: "Mistral Nemo (open)",
          contextWindow: 131072,
          maxOutput: 4096,
        },
      ],
    },
    auth: [
      {
        type: "bearer",
        envVar: "MISTRAL_API_KEY",
        headerName: "Authorization",
        label: "Mistral AI API Key",
      },
    ],
  });

  // Google Gemini — via OpenAI-compatible endpoint
  api.registerProvider({
    id: "google",
    label: "Google Gemini",
    docsPath: "https://ai.google.dev/docs",
    aliases: ["gemini", "google-gemini"],
    envVars: ["GOOGLE_API_KEY"],
    models: {
      chat: [
        {
          id: "gemini-2.0-flash",
          label: "Gemini 2.0 Flash",
          contextWindow: 1048576,
          maxOutput: 8192,
        },
        {
          id: "gemini-2.0-flash-thinking-exp",
          label: "Gemini 2.0 Flash Thinking (exp)",
          contextWindow: 1048576,
          maxOutput: 65536,
        },
        {
          id: "gemini-1.5-pro",
          label: "Gemini 1.5 Pro",
          contextWindow: 2097152,
          maxOutput: 8192,
        },
        {
          id: "gemini-1.5-flash",
          label: "Gemini 1.5 Flash",
          contextWindow: 1048576,
          maxOutput: 8192,
        },
      ],
    },
    auth: [
      {
        type: "bearer",
        envVar: "GOOGLE_API_KEY",
        headerName: "Authorization",
        label: "Google AI API Key",
      },
    ],
  });

  const bannerEndpoint = onboardCfg?.endpointType ?? "build.nvidia.com";
  const bannerModel = onboardCfg?.model ?? "nvidia/nemotron-3-super-120b-a12b";

  api.logger.info("");
  api.logger.info("  ┌─────────────────────────────────────────────────────┐");
  api.logger.info("  │  NemoClaw registered                                │");
  api.logger.info("  │                                                     │");
  api.logger.info(`  │  Endpoint:  ${bannerEndpoint.padEnd(40)}│`);
  api.logger.info(`  │  Model:     ${bannerModel.padEnd(40)}│`);
  api.logger.info("  │  Commands:  openclaw nemoclaw <command>             │");
  api.logger.info("  └─────────────────────────────────────────────────────┘");
  api.logger.info("");
}
