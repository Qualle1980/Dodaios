import { describe, expect, it } from "vitest";

import {
  agentToolView,
  cachedTokens,
  formatTokens,
  inputSummary,
  inputTokens,
  outputTokens,
  shortenPath,
} from "@/components/agentchat/toolView";

describe("agentToolView", () => {
  it("keeps the agent's own tool name — the row must match the log", () => {
    expect(agentToolView("PowerShell", { command: "Get-ChildItem" })).toMatchObject({
      label: "PowerShell",
      family: "shell",
      summary: "Get-ChildItem",
    });
    expect(agentToolView("ToolSearch", { query: "chrome tabs" })).toMatchObject({
      label: "ToolSearch",
      family: "tools",
      summary: "chrome tabs",
    });
    // A Grep shows WHAT was searched for, not the folder it looked in — the
    // path alone reads as the wrong call next to the agent's log.
    const grep = agentToolView("Grep", {
      path: "C:\\Users\\Administrator\\Desktop",
      pattern: "permissionMode",
    });
    expect(grep.label).toBe("Grep");
    expect(grep.summary).toBe("permissionMode");
    expect(agentToolView("Read", { file_path: "a.py", pattern: "x" }).summary).toBe("a.py");
    expect(agentToolView("Read", { file_path: "a.py" }).family).toBe("read");
    expect(agentToolView("MultiEdit", {}).family).toBe("edit");
    expect(agentToolView("Bash", { command: "ls -la\nsecond line" }).summary).toBe("ls -la");
    expect(agentToolView("TodoWrite", {}).family).toBe("plan");
    expect(agentToolView("Task", { prompt: "go" }).family).toBe("agent");
  });

  it("reads the other CLIs' tool names and argument spellings the same way", () => {
    // Grok Build
    expect(agentToolView("read_file", { target_file: "a.py" })).toMatchObject({
      family: "read",
      summary: "a.py",
    });
    expect(agentToolView("search_replace", { target_file: "a.py" }).family).toBe("edit");
    expect(agentToolView("list_dir", { target_directory: "src" }).summary).toBe("src");
    // OpenCode
    expect(agentToolView("read", { filePath: "run.bat" }).summary).toBe("run.bat");
    // Antigravity
    expect(agentToolView("find_by_name", { Pattern: "*.png", SearchDirectory: "src" })).toMatchObject({
      family: "search",
      summary: "*.png",
    });
    expect(agentToolView("run_command", { CommandLine: "npm test" })).toMatchObject({
      family: "shell",
      summary: "npm test",
    });
    expect(agentToolView("write_to_file", { TargetFile: "x.md" }).family).toBe("write");
    expect(agentToolView("view_file", { AbsolutePath: "C:/w/a.png" }).family).toBe("read");
    // Kimi
    expect(agentToolView("ReadFile", { path: "NOTES.md" })).toMatchObject({
      family: "read",
      summary: "NOTES.md",
    });
  });

  it("names an MCP call after its server and wears the vendor mark when there is one", () => {
    const gh = agentToolView("mcp__github__create_issue", { title: "Bug" });
    expect(gh.family).toBe("mcp");
    expect(gh.label).toBe("GitHub · Create Issue");
    expect(gh.logoUrl).toBeTruthy();
    const slash = agentToolView("blender/get_scene_info", {});
    expect(slash.family).toBe("mcp");
    expect(slash.label).toContain("Blender");
  });

  it("falls back to a wrench for a tool nobody knows, never to nothing", () => {
    const odd = agentToolView("frobnicate", { thing: "x" });
    expect(odd.label).toBe("frobnicate");
    expect(odd.family).toBe("other");
    expect(agentToolView("", {}).label).toBe("tool");
  });
});

describe("inputSummary", () => {
  it("prefers the human field and keeps it to one line", () => {
    expect(inputSummary({ command: "pytest -q" })).toBe("pytest -q");
    expect(inputSummary({ nothing: 3, note: "hello" })).toBe("hello");
    expect(inputSummary("plain string")).toBe("plain string");
    expect(inputSummary(null)).toBe("");
    expect(inputSummary({ command: "x".repeat(300) })).toHaveLength(200);
  });
});

describe("token formatting", () => {
  it("reads like the Claude CLI: 4.8k, 1.2M", () => {
    expect(formatTokens(219)).toBe("219");
    expect(formatTokens(4800)).toBe("4.8k");
    expect(formatTokens(12000)).toBe("12k");
    expect(formatTokens(1_200_000)).toBe("1.2M");
    expect(formatTokens(-5)).toBe("0");
  });

  it("reads either usage shape", () => {
    expect(outputTokens({ output_tokens: 5 })).toBe(5);
    expect(outputTokens({ output: 7 })).toBe(7);
    expect(outputTokens(null)).toBeNull();
    expect(outputTokens({})).toBeNull();
  });

  /*
   * Input is the bill, and it was the one number no receipt showed.
   *
   * Measured live 2026-10-31: twelve chat turns on a plan seat ran
   * 16_356-17_156 input tokens against 68-179 output, and later turns reached
   * 32_956 input. The receipt printed the output alone and no price, because a
   * plan seat prices at $0 and the old receipt was gated on costUsd > 0. So the
   * surface spending the plan window was the only one that never said so.
   */
  it("reads the input side, in every shape a brain sends", () => {
    expect(inputTokens({ input_tokens: 16356 })).toBe(16356);
    expect(inputTokens({ input: 12 })).toBe(12);
    expect(inputTokens({ prompt_tokens: 9 })).toBe(9);
    expect(inputTokens(null)).toBeNull();
    expect(inputTokens({})).toBeNull();
    expect(inputTokens({ output_tokens: 104 })).toBeNull();
  });

  it("reads the cached share, so a cache that stopped working is visible", () => {
    expect(cachedTokens({ cache_hit_tokens: 33536 })).toBe(33536);
    expect(cachedTokens({ cached_tokens: 1 })).toBe(1);
    expect(cachedTokens({ cache_read_input_tokens: 2 })).toBe(2);
    expect(cachedTokens({ input_tokens: 500 })).toBeNull();
    expect(cachedTokens(null)).toBeNull();
  });

  it("the live turn shape: input dwarfs output", () => {
    const usage = { input_tokens: 16356, output_tokens: 179, cache_hit_tokens: 0 };
    expect(inputTokens(usage)).toBe(16356);
    expect(outputTokens(usage)).toBe(179);
    expect(formatTokens(inputTokens(usage)!)).toBe("16.4k");
  });
});

/*
 * A path on a row.
 *
 * Every file in one workspace shares its first six or seven segments, so a
 * row printing the whole of an absolute path spends its width saying where
 * the project lives and runs out before saying which file (live check,
 * 2026-08-27).
 */
describe("shortenPath", () => {
  it("keeps the three segments that tell a row apart from its neighbours", () => {
    expect(
      shortenPath("C:\\Users\\Someone\\Desktop\\Project\\src\\brain\\healthcheck.py"),
    ).toBe("…/src/brain/healthcheck.py");
    expect(shortenPath("/home/someone/work/repo/src/app.ts")).toBe("…/repo/src/app.ts");
  });

  it("leaves a path that is already short exactly as it is", () => {
    expect(shortenPath("/etc/hosts")).toBe("/etc/hosts");
    expect(shortenPath("src/app.ts")).toBe("src/app.ts");
  });

  it("does not touch things that merely contain a slash", () => {
    expect(shortenPath("a/b or c/d — either way")).toBe("a/b or c/d — either way");
    expect(shortenPath("rg -n 'foo/bar' src")).toBe("rg -n 'foo/bar' src");
  });

  it("reaches the row through the summary, whichever way the agent spells it", () => {
    expect(inputSummary({ file_path: "C:\\a\\b\\c\\d\\e.ts" }, "edit")).toBe("…/c/d/e.ts");
  });
});
