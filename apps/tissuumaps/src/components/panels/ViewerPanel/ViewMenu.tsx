import { EyeIcon } from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useProjectStore } from "@/stores/project";

export type ViewMenuProps = { className?: string };

export function ViewMenu({ className }: ViewMenuProps) {
  const osOptions = useProjectStore((state) => state.osOptions);
  const setOSOptions = useProjectStore((state) => state.setOSOptions);

  const navigatorVisible = osOptions.viewerOptions.showNavigator !== false;

  return (
    // not modal: the viewer stays usable while the menu is open
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger
        render={<IconButton label="View" className={className} />}
      >
        <EyeIcon />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuCheckboxItem
          checked={navigatorVisible}
          onCheckedChange={(checked) => {
            setOSOptions({
              ...osOptions,
              viewerOptions: {
                ...osOptions.viewerOptions,
                showNavigator: checked,
              },
            });
          }}
        >
          Navigator
        </DropdownMenuCheckboxItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
