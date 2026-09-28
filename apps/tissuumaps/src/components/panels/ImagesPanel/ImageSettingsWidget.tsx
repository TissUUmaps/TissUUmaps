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
import { useControlled } from "@/hooks/useControlled";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import { ImageSettingsCategory } from "./category";

export type ImageSettingsWidgetProps = {
  image: Image;
  activeCategory?: ImageSettingsCategory | null;
  onActiveCategoryChange?: (
    newActiveCategory: ImageSettingsCategory | null,
  ) => void;
  className?: string;
};

export function ImageSettingsWidget({
  image,
  activeCategory: controlledActiveCategory,
  onActiveCategoryChange: setControlledActiveCategory,
  className,
}: ImageSettingsWidgetProps) {
  const updateImage = useProjectStore((state) => state.updateImage);

  const [activeCategory, setActiveCategory] = useControlled(
    controlledActiveCategory,
    setControlledActiveCategory,
    null,
  );

  return (
    <Fieldset
      className={cn("flex flex-col gap-y-2 border rounded-md p-2", className)}
    >
      <FieldsetLegend className="font-medium text-foreground">
        Settings
      </FieldsetLegend>
      <Accordion
        value={activeCategory !== null ? [activeCategory] : []}
        onValueChange={(value) =>
          setActiveCategory(
            value.length > 0 ? (value[0] as ImageSettingsCategory) : null,
          )
        }
      >
        <AccordionItem value={ImageSettingsCategory.general}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>General</AccordionTrigger>
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <GeneralImageSettingsWidget image={image} />
          </AccordionPanel>
        </AccordionItem>
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

type GeneralImageSettingsWidgetProps = {
  image: Image;
  className?: string;
};

function GeneralImageSettingsWidget({
  image,
  className,
}: GeneralImageSettingsWidgetProps) {
  const layers = useProjectStore((state) => state.layers);
  const updateImage = useProjectStore((state) => state.updateImage);

  return (
    <div className={className}>
      <Field>
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
