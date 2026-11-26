import { execFileSync } from "node:child_process";
import { appendFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import semanticRelease from "semantic-release";

const releasesPath = `repos/${process.env.GITHUB_REPOSITORY}/releases`;

async function github(path, method = "GET", body, contentType = "application/json") {
  const response = await fetch(new URL(path, `${process.env.GITHUB_API_URL}/`), {
    method,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${process.env.GITHUB_TOKEN}`,
      "Content-Type": contentType,
    },
    body,
  });
  if (!response.ok) {
    throw new Error(`GitHub ${method} ${path} failed (${response.status}): ${await response.text()}`);
  }
  if (response.status !== 204) {
    return response.json();
  }
}

async function listReleases() {
  const releases = [];
  for (let page = 1; ; page++) {
    const batch = await github(`${releasesPath}?per_page=100&page=${page}`);
    releases.push(...batch);
    if (batch.length < 100) {
      return releases;
    }
  }
}

function report(message) {
  console.log(message);
  if (process.env.GITHUB_STEP_SUMMARY) {
    appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${message}\n`);
  }
}

async function calculateRelease() {
  const result = await semanticRelease({
    branches: ["main"],
    dryRun: true,
    plugins: [
      ["@semantic-release/commit-analyzer", { preset: "conventionalcommits" }],
      ["@semantic-release/release-notes-generator", { preset: "conventionalcommits" }],
    ],
  });
  return result?.nextRelease;
}

async function createDraft() {
  const sha = process.env.GITHUB_SHA;
  const releases = await listReleases();
  const drafts = releases.filter(({ draft, tag_name }) => draft && /^v\d+\.\d+\.\d+$/.test(tag_name));
  let release = drafts.find(({ target_commitish }) => target_commitish === sha);
  const candidate = release
    ? {
        gitHead: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
        gitTag: release.tag_name,
        version: release.tag_name.slice(1),
      }
    : await calculateRelease();

  if (!candidate) {
    report("No release candidate for this run.");
    return;
  }

  const { gitHead, gitTag, notes, version } = candidate;
  if (gitHead !== sha) {
    throw new Error(`Release candidate ${gitHead} does not match the workflow commit ${sha}.`);
  }

  const published = releases.find(({ draft, tag_name }) => !draft && tag_name === gitTag);
  if (published) {
    throw new Error(`${gitTag} already has a published release: ${published.html_url}`);
  }
  const conflictingDraft = drafts.find((draft) => draft !== release);
  if (conflictingDraft) {
    throw new Error(`Publish or delete the existing draft before running this workflow again: ${conflictingDraft.html_url}`);
  }
  const remoteTag = execFileSync("git", ["ls-remote", "--tags", "origin", `refs/tags/${gitTag}`], { encoding: "utf8" });
  if (remoteTag.trim()) {
    throw new Error(`Tag ${gitTag} already exists on origin.`);
  }

  const assetName = `APMI-Tools-${gitTag}.zip`;
  const existingAsset = release?.assets.find(({ name }) => name === assetName);
  if (existingAsset?.state === "uploaded") {
    report(`Draft release already exists with ${assetName}: ${release.html_url}`);
    return;
  }

  const directory = mkdtempSync(join(tmpdir(), "apmi-release-"));
  try {
    const archive = join(directory, assetName);
    const packageScript = fileURLToPath(new URL("./package-addon.py", import.meta.url));
    execFileSync(process.platform === "win32" ? "python" : "python3", [packageScript, version, archive], { stdio: "inherit" });

    if (!release) {
      release = await github(releasesPath, "POST", JSON.stringify({
        tag_name: gitTag,
        target_commitish: gitHead,
        name: gitTag,
        body: notes,
        draft: true,
      }));
    }

    if (existingAsset?.state === "starter") {
      await github(`${releasesPath}/assets/${existingAsset.id}`, "DELETE");
    }
    const uploadUrl = new URL(release.upload_url.split("{")[0]);
    uploadUrl.searchParams.set("name", assetName);
    await github(uploadUrl, "POST", readFileSync(archive), "application/zip");
    report(`Draft release ${gitTag} with ${assetName}: ${release.html_url}`);
  } catch (error) {
    if (release) {
      console.error(`Draft release: ${release.html_url}`);
    }
    throw error;
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

try {
  await createDraft();
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
