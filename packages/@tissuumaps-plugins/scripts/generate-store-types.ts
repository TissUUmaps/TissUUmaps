/// <reference types="node" />
// Generates src/ai-chat/store-types.generated.ts: the types of the app state
// of TissUUmaps that the AI chat tells the model about (the project store,
// the model of the project, and the parts of the app store that only change
// the view), resolved by the TypeScript compiler into plain property lists,
// each property with the first sentence of its comment.
//
// Run it after changing the types of @tissuumaps/core, in this package:
//   pnpm run generate:store-types
// A test checks that the generated file is up to date.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import ts from "typescript";

/**
 * The types the model is told about, in this order, optionally only with some
 * of their properties
 */
const entries: { name: string; pick?: string[] }[] = [
  { name: "ProjectStoreState" },
  { name: "ProjectStoreActions" },
  { name: "Layer" },
  { name: "Image" },
  { name: "ImageChannel" },
  { name: "ImageChannelViewMode" },
  { name: "Labels" },
  { name: "Points" },
  { name: "Shapes" },
  { name: "Table" },
  { name: "GroupValueMap" },
  { name: "ColorConfig" },
  { name: "VisibilityConfig" },
  { name: "OpacityConfig" },
  { name: "SizeConfig" },
  { name: "MarkerConfig" },
  { name: "TableColumnRef" },
  { name: "SimilarityTransform" },
  { name: "Color" },
  { name: "Marker" },
  { name: "CoordinateSpace" },
  {
    name: "AppStoreState",
    pick: [
      "interactionMode",
      "imageChannelPreview",
      "highlightedItemGroup",
      "activePanelId",
      "expandedImageIds",
      "expandedLabelsIds",
      "expandedPointsIds",
      "expandedShapesIds",
      "expandedTableIds",
    ],
  },
  {
    name: "AppStoreActions",
    pick: [
      "setInteractionMode",
      "setImageChannelPreview",
      "setHighlightedItemGroup",
      "setActivePanelId",
      "setExpandedImageIds",
      "setExpandedLabelsIds",
      "setExpandedPointsIds",
      "setExpandedShapesIds",
      "setExpandedTableIds",
    ],
  },
  { name: "InteractionMode" },
  { name: "ImageChannelPreview" },
  { name: "HighlightedItemGroup" },
];

/** The types written by name where they are used, since they have an entry */
const named = new Set(entries.map(({ name }) => name));

/**
 * Helper types whose name means nothing to the model: what they stand for is
 * written out instead
 */
const writtenOut = new Set([
  "Config",
  "ConstantConfig",
  "FromConfig",
  "GroupByConfig",
  "RandomConfig",
]);

/** The types whose property names are also listed, to check changes */
const listed = ["Layer", "Image", "ImageChannel", "Labels", "Points", "Shapes"];

/**
 * Generates the text of the types, and the property names of some of them
 *
 * @returns The text, one line per type or per property of a type, and the
 * property names by type
 */
export function generateStoreTypes(): {
  text: string;
  properties: Record<string, string[]>;
} {
  const core = fileURLToPath(
    new URL("../../@tissuumaps-core/", import.meta.url),
  );
  const configFile = ts.readConfigFile(`${core}tsconfig.src.json`, (path) =>
    ts.sys.readFile(path),
  );
  const config = ts.parseJsonConfigFileContent(configFile.config, ts.sys, core);
  const program = ts.createProgram(config.fileNames, config.options);
  const checker = program.getTypeChecker();
  const index = program.getSourceFile(`${core}src/index.ts`);
  const moduleSymbol =
    index === undefined ? undefined : checker.getSymbolAtLocation(index);
  if (moduleSymbol === undefined) {
    throw new Error("cannot read the exports of @tissuumaps/core");
  }
  const exported = new Map(
    checker
      .getExportsOfModule(moduleSymbol)
      .map((symbol) => [
        symbol.name,
        symbol.flags & ts.SymbolFlags.Alias
          ? checker.getAliasedSymbol(symbol)
          : symbol,
      ]),
  );
  const flags =
    ts.TypeFormatFlags.NoTruncation |
    ts.TypeFormatFlags.UseAliasDefinedOutsideCurrentScope;

  /** Removes the spaces that the compiler puts around punctuation */
  function compact(text: string): string {
    return text
      .replace(/\s*([{}()[\];:,|&<>=?])\s*/g, "$1")
      .replace(/;}/g, "}");
  }

  /**
   * The first sentence of a comment, without {@link} markup, or "" if it only
   * speaks about how the type is built (e.g. "A RawLabels with labelsDefaults
   * applied")
   */
  function firstSentence(symbol: ts.Symbol): string {
    return firstSentenceOf(
      ts.displayPartsToString(symbol.getDocumentationComment(checker)),
    );
  }

  /**
   * The first sentence of the comment of a type, not of a value with the same
   * name (e.g. the Marker object next to the Marker type)
   */
  function typeComment(symbol: ts.Symbol): string {
    const declaration = symbol.declarations?.find(
      (node) =>
        ts.isTypeAliasDeclaration(node) || ts.isInterfaceDeclaration(node),
    );
    const [doc] =
      declaration === undefined
        ? []
        : ts.getJSDocCommentsAndTags(declaration).filter(ts.isJSDoc);
    return firstSentenceOf(ts.getTextOfJSDocComment(doc?.comment) ?? "");
  }

  /** See {@link firstSentence} */
  function firstSentenceOf(comment: string): string {
    const [paragraph = ""] = comment
      // the same on every computer, whatever its line endings
      .replaceAll("\r", "")
      .split(/\n\s*\n/);
    const text = paragraph
      .replace(/\{@link\s+([^}|\s]+)\s*(\|[^}]*)?\}/g, "$1")
      .replace(/\s*\n\s*/g, " ")
      .trim();
    // a sentence ends before a capital letter, not in "i.e. the"
    const [sentence = ""] = text.split(/(?<=[.!?])\s+(?=[A-Z])/);
    return /\bRaw\w+ .*defaults applied/i.test(sentence)
      ? ""
      : sentence.replace(/\.$/, "");
  }

  /** Whether a type comes from outside @tissuumaps/core, e.g. Map or Promise */
  function isForeign(type: ts.Type): boolean {
    const declarations = (type.aliasSymbol ?? type.getSymbol())?.declarations;
    return (declarations ?? []).some(
      (declaration) =>
        !declaration
          .getSourceFile()
          .fileName.startsWith(core.replaceAll("\\", "/")),
    );
  }

  /** Writes a type: by name if it has an entry, or else written out */
  function typeText(type: ts.Type): string {
    const alias = type.aliasSymbol?.name;
    if (alias === "NonNullable" && type.aliasTypeArguments?.[0] !== undefined) {
      return typeText(type.aliasTypeArguments[0]);
    }
    const name = alias ?? type.getSymbol()?.name;
    if (name !== undefined && named.has(name)) {
      const args = type.aliasTypeArguments ?? [];
      return args.length === 0
        ? name
        : `${name}<${args.map(typeText).join(",")}>`;
    }
    if (type.flags & ts.TypeFlags.Boolean) {
      return "boolean";
    }
    if (type.isUnion()) {
      return unionText(type.types);
    }
    if (checker.isArrayType(type)) {
      const [element] = checker.getTypeArguments(type as ts.TypeReference);
      const text = element === undefined ? "unknown" : typeText(element);
      return text.includes("|") ? `(${text})[]` : `${text}[]`;
    }
    const [signature] = type.getCallSignatures();
    if (signature !== undefined) {
      return `${parametersText(signature)}=>${typeText(signature.getReturnType())}`;
    }
    const isObject = type.flags & ts.TypeFlags.Object || type.isIntersection();
    if (
      isObject &&
      !checker.isTupleType(type) &&
      !isForeign(type) &&
      (alias === undefined || writtenOut.has(alias))
    ) {
      return `{${membersOf(type).join(";")}}`;
    }
    return compact(checker.typeToString(type, undefined, flags));
  }

  /** Writes the members of a union, e.g. "null|string" */
  function unionText(types: ts.Type[]): string {
    const members = types.map(typeText);
    // a boolean is the union of true and false
    const merged =
      members.includes("true") && members.includes("false")
        ? [
            ...members.filter(
              (member) => member !== "true" && member !== "false",
            ),
            "boolean",
          ]
        : members;
    return [...new Set(merged)].join("|");
  }

  /** The type of an optional property or parameter, without `undefined` */
  function definedText(type: ts.Type, optional: boolean): string {
    if (!optional || !type.isUnion()) {
      return typeText(type);
    }
    const defined = type.types.filter(
      (member) => !(member.flags & ts.TypeFlags.Undefined),
    );
    // the same members as a named union, e.g. ColorConfig, keep its name
    const nonNullable = checker.getNonNullableType(type);
    if (
      !defined.some((member) => member.flags & ts.TypeFlags.Null) &&
      nonNullable.aliasSymbol !== undefined
    ) {
      return typeText(nonNullable);
    }
    return unionText(defined);
  }

  /** The parameters of a function, e.g. "(imageId:string,updates:...)" */
  function parametersText(signature: ts.Signature): string {
    const parameters = signature.getParameters().map((parameter) => {
      const declaration = parameter.valueDeclaration;
      const optional =
        declaration !== undefined &&
        ts.isParameter(declaration) &&
        (declaration.questionToken !== undefined ||
          declaration.initializer !== undefined);
      const type = checker.getTypeOfSymbol(parameter);
      return `${parameter.name}${optional ? "?" : ""}:${definedText(type, optional)}`;
    });
    return `(${parameters.join(",")})`;
  }

  /** The properties of an object type, e.g. ["id:string", "name?:string"] */
  function membersOf(type: ts.Type, pick?: string[]): string[] {
    const properties = checker
      .getPropertiesOfType(type)
      .filter((property) => pick === undefined || pick.includes(property.name));
    const lines = properties.map((property) => {
      const propertyType = checker.getTypeOfSymbol(property);
      const optional = (property.flags & ts.SymbolFlags.Optional) !== 0;
      const [signature] = propertyType.getCallSignatures();
      // actions are written as methods
      return signature !== undefined && !optional
        ? `${property.name}${parametersText(signature)}`
        : `${property.name}${optional ? "?" : ""}:${definedText(propertyType, optional)}`;
    });
    const indexes = checker
      .getIndexInfosOfType(type)
      .map(
        (info) =>
          `[key:${checker.typeToString(info.keyType)}]:${typeText(info.type)}`,
      );
    return [...indexes, ...lines];
  }

  /** The same, each property on its own line with its comment */
  function documentedMembersOf(type: ts.Type, pick?: string[]): string[] {
    const properties = checker
      .getPropertiesOfType(type)
      .filter((property) => pick === undefined || pick.includes(property.name));
    const lines = membersOf(type, pick);
    // index signatures come first and have no comment
    const offset = lines.length - properties.length;
    return lines.map((line, position) => {
      const property = properties[position - offset];
      const comment = property === undefined ? "" : firstSentence(property);
      return comment === "" ? line : `${line} // ${comment}`;
    });
  }

  const text = entries.map(({ name, pick }) => {
    const symbol = exported.get(name);
    if (symbol === undefined) {
      throw new Error(`@tissuumaps/core does not export ${name}`);
    }
    const type = checker.getDeclaredTypeOfSymbol(symbol);
    const parameters = (type.aliasTypeArguments ?? [])
      .map((parameter) => checker.typeToString(parameter))
      .join(",");
    const head = `type ${name}${parameters === "" ? "" : `<${parameters}>`}=`;
    let comment = typeComment(symbol);
    // numbers named by an object with the same name, e.g. Marker.Disc = 6
    const values =
      symbol.valueDeclaration !== undefined
        ? checker.getPropertiesOfType(checker.getTypeOfSymbol(symbol))
        : [];
    if (
      values.length > 0 &&
      type.isUnion() &&
      type.types.every((member) => member.isNumberLiteral())
    ) {
      const names = values
        .map(
          (value) =>
            `${checker.typeToString(checker.getTypeOfSymbol(value))} ${value.name}`,
        )
        .join(", ");
      comment = `${comment}: ${names}`;
    }
    const note = `${comment === "" ? "" : ` // ${comment}`}${pick === undefined ? "" : " (only some)"}`;
    if (type.isUnion()) {
      // a union of objects, e.g. ColorConfig: one line per kind
      if (
        type.types.some(
          (member) =>
            member.flags & (ts.TypeFlags.Object | ts.TypeFlags.Intersection),
        )
      ) {
        return [
          `${head}${note}`,
          ...type.types.map((member) => `|{${membersOf(member).join(";")}}`),
        ].join("\n");
      }
      // e.g. Marker: its values
      return `${head}${type.types.map(typeText).join("|")}${note}`;
    }
    if (type.flags & ts.TypeFlags.Object || type.isIntersection()) {
      return [`${head}{${note}`, ...documentedMembersOf(type, pick), "}"].join(
        "\n",
      );
    }
    return `${head}${typeText(type)}${note}`;
  });
  const properties = Object.fromEntries(
    listed.map((name) => {
      const symbol = exported.get(name);
      if (symbol === undefined) {
        throw new Error(`@tissuumaps/core does not export ${name}`);
      }
      const type = checker.getDeclaredTypeOfSymbol(symbol);
      return [
        name,
        checker.getPropertiesOfType(type).map((property) => property.name),
      ];
    }),
  );
  return { text: text.join("\n"), properties };
}

/**
 * Writes the text in quotes, as Prettier would: double quotes, unless the
 * text has more of them than single quotes
 *
 * @param text - The text
 * @returns The string literal
 */
function quote(text: string): string {
  const double = text.split('"').length;
  const single = text.split("'").length;
  return double > single
    ? `'${text.replaceAll("\\", "\\\\").replaceAll("'", "\\'")}'`
    : JSON.stringify(text);
}

/** Writes the generated file, formatted by Prettier */
async function writeStoreTypes(): Promise<void> {
  const prettier = await import("prettier");
  const file = fileURLToPath(
    new URL("../src/ai-chat/store-types.generated.ts", import.meta.url),
  );
  const { text, properties } = generateStoreTypes();
  const lines = text.split("\n");
  const source = [
    "// Generated by scripts/generate-store-types.ts from the types of",
    "// @tissuumaps/core: do not edit by hand. After changing those types, run",
    "// `pnpm run generate:store-types` in packages/@tissuumaps-plugins.",
    "",
    "/**",
    " * The types of the app state of TissUUmaps that the AI chat tells the model",
    " * about: the project store, the model of the project, and the parts of the",
    " * app store that only change the view, resolved into plain property lists",
    " */",
    `export const storeTypes = [${lines.map(quote).join(",")}].join("\\n");`,
    "",
    "/**",
    " * The property names of some of those types, by type, to check what the",
    " * model asks to change",
    " */",
    `export const storeProperties: Record<string, string[]> = ${JSON.stringify(properties)};`,
    "",
  ].join("\n");
  const options = await prettier.resolveConfig(file);
  writeFileSync(
    file,
    await prettier.format(source, { ...options, filepath: file }),
  );
}

// when run with node, not when a test imports it
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await writeStoreTypes();
}
