import type * as Preset from "@docusaurus/preset-classic";
import type { Config } from "@docusaurus/types";
import { themes as prismThemes } from "prism-react-renderer";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

// The documentation is deployed under `<application>/docs/`, so the "Live"
// links point at that application, or at the development server when
// building locally (absolute, since Docusaurus treats paths as internal links)
const url = "https://tissuumaps.github.io";
const baseUrl = process.env.DOCUSAURUS_BASE_URL || "/";
if (baseUrl !== "/" && !baseUrl.endsWith("/docs/")) {
  throw new Error(`DOCUSAURUS_BASE_URL must end in "/docs/": ${baseUrl}`);
}
const appUrl =
  baseUrl === "/"
    ? "http://localhost:5173/"
    : url + baseUrl.replace(/docs\/$/, "");

const config: Config = {
  title: "TissUUmaps",
  tagline: "Spatial Biology Visualization",
  favicon: "img/favicon.ico",

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url,
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl,

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "TissUUmaps", // Usually your GitHub org/user name.
  projectName: "TissUUmaps", // Usually your repo name.

  onBrokenLinks: "throw",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            "https://github.com/TissUUmaps/TissUUmaps/tree/main/apps/docs/",
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: "TissUUmaps",
      logo: {
        alt: "TissUUmaps Logo",
        src: "img/logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "docsSidebar",
          position: "left",
          label: "Documentation",
        },
        {
          type: "docSidebar",
          sidebarId: "apiSidebar",
          position: "left",
          label: "Packages",
        },
        {
          href: "https://github.com/TissUUmaps/TissUUmaps/",
          label: "GitHub",
          position: "right",
        },
        {
          href: appUrl,
          label: "Live",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "TissUUmaps",
          items: [
            {
              label: "Documentation",
              to: "/docs/intro",
            },
            {
              label: "Packages",
              to: "/docs/api",
            },
          ],
        },
        {
          title: "Community",
          items: [
            {
              label: "SciLifeLab BioImage Informatics Unit",
              href: "https://www.scilifelab.se/units/bioimage-informatics/",
            },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/TissUUmaps/TissUUmaps/",
            },
            {
              href: appUrl,
              label: "Live",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} TissUUmaps Core Team. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,

  plugins: [
    [
      "docusaurus-plugin-typedoc",
      {
        name: "API",
        entryPoints: ["../../packages/*"],
        entryPointStrategy: "packages",
        packageOptions: {
          entryPoints: ["src/index.ts"],
        },
        sidebar: {
          typescript: true,
        },
        readme: "none",
      },
    ],
  ],

  // mermaid diagrams
  // https://docusaurus.io/docs/markdown-features/diagrams
  markdown: {
    mermaid: true,
  },
  themes: ["@docusaurus/theme-mermaid"],
};

export default config;
