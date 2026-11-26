import type { Image } from "@tissuumaps/core";

import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  AccordionTrigger,
  AccordionTriggerRightDownIcon,
} from "@/components/common/accordion";
import { Fieldset, FieldsetLegend } from "@/components/common/fieldset";
import { LayerField } from "@/components/controls/LayerField";
import { TransformSettingsWidget } from "@/components/widgets/TransformSettingsWidget";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import { ImageSettingsCategory } from "./category";

export type ImageSettingsWidgetProps = {
  image: Image;
  className?: string;
};

export function ImageSettingsWidget({
  image,
  className,
}: ImageSettingsWidgetProps) {
  const updateImage = useProjectStore((state) => state.updateImage);

  return (
    <Fieldset
      className={cn("flex flex-col gap-y-2 border rounded-md p-2", className)}
    >
      <FieldsetLegend className="font-medium text-foreground">
        Settings
      </FieldsetLegend>
      <LayerField
        value={image.layer}
        onValueChange={(layer) => updateImage(image.id, { layer })}
      />
      <Accordion>
        <AccordionItem value={ImageSettingsCategory.transform}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Transform</AccordionTrigger>
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <TransformSettingsWidget
              transform={image.transform}
              onTransformChange={(transform) =>
                updateImage(image.id, { transform })
              }
            />
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
    </Fieldset>
  );
}
