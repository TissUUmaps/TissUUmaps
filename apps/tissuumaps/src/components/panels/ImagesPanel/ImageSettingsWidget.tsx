import type { Image } from "@tissuumaps/core";

import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  AccordionTrigger,
  AccordionTriggerRightDownIcon,
} from "@/components/common/accordion";
import { Field, FieldLabel } from "@/components/common/field";
import { Fieldset, FieldsetLegend } from "@/components/common/fieldset";
import { SimpleSelect } from "@/components/common/simple-select";
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
      <ImageLayerSettingsWidget image={image} />
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

type ImageLayerSettingsWidgetProps = {
  image: Image;
  className?: string;
};

function ImageLayerSettingsWidget({
  image,
  className,
}: ImageLayerSettingsWidgetProps) {
  const layers = useProjectStore((state) => state.layers);
  const updateImage = useProjectStore((state) => state.updateImage);

  return (
    <div className={className}>
      <Field className="flex flex-row items-center gap-x-2">
        <FieldLabel>Layer</FieldLabel>
        <SimpleSelect
          items={layers}
          itemLabel={(l) => l.name}
          itemValue={(l) => l.id}
          value={image.layer}
          onValueChange={(value) => {
            if (value !== null) {
              updateImage(image.id, { layer: value });
            }
          }}
        />
      </Field>
    </div>
  );
}
