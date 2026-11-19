import { describe, expect, it } from "vitest";

import { skillPackageName } from "./python-imports";
import { loadSkill } from "./skill-tools";
import type { Skill } from "./skills";
import type { ToolContext } from "./types";

/** A skill with some files */
function skillWith(name: string, paths: string[]): Skill {
  return {
    name,
    description: "Finds regions.",
    metadata: {},
    instructions: "Use it.",
    skillMd: "",
    files: new Map(paths.map((path) => [path, new Blob(["x"])])),
    warnings: [],
  };
}

describe("skills that are Python packages", () => {
  it("names the package of a skill whose folder has __init__.py", () => {
    expect(skillPackageName(skillWith("points2regions", ["__init__.py"]))).toBe(
      "points2regions",
    );
    expect(skillPackageName(skillWith("my-tool", ["__init__.py"]))).toBe(
      "my_tool",
    );
    expect(skillPackageName(skillWith("2d-maps", ["__init__.py"]))).toBe(
      "_2d_maps",
    );
    expect(
      skillPackageName(skillWith("anhood", ["anhood/__init__.py"])),
    ).toBeUndefined();
  });

  it("tells the model, with load_skill, how to import it", () => {
    const skill = skillWith("points2regions", [
      "__init__.py",
      "_points2regions.py",
    ]);
    const text = loadSkill({ name: "points2regions" }, {
      skills: [skill],
    } as unknown as ToolContext);
    expect(text).toContain(
      "Its folder is a Python package (it has __init__.py): in Python with this skill, import it by its name, e.g. import points2regions",
    );
    const plain = loadSkill({ name: "anhood" }, {
      skills: [skillWith("anhood", ["anhood/__init__.py"])],
    } as unknown as ToolContext);
    expect(plain).not.toContain("Python package");
  });
});
