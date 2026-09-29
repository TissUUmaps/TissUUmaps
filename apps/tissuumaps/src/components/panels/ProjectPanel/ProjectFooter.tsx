import logoUrl from "@/assets/logo.svg";
import { cn } from "@/lib/utils";

export type ProjectFooterProps = {
  className?: string;
};

export function ProjectFooter({ className }: ProjectFooterProps) {
  return (
    <footer
      className={cn(
        "text-muted-foreground flex items-center gap-2 border-t px-2 pt-2 text-xs",
        className,
      )}
    >
      <img src={logoUrl} alt="" className="h-7" />
      <span className="text-foreground text-sm font-semibold">TissUUmaps</span>
      <span>{__APP_VERSION__}</span>
      <nav className="ml-auto flex gap-3">
        <a
          href="https://tissuumaps.github.io/TissUUmaps/docs/"
          target="_blank"
          rel="noreferrer"
          className="hover:text-foreground"
        >
          Docs
        </a>
        <a
          href={__APP_REPOSITORY_URL__}
          target="_blank"
          rel="noreferrer"
          className="hover:text-foreground"
        >
          GitHub
        </a>
      </nav>
    </footer>
  );
}
