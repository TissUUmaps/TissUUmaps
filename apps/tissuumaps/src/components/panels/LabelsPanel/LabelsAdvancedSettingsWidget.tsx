import {
  type Labels,
  defaultLabelColor,
  defaultLabelOpacity,
  defaultLabelVisibility,
} from "@tissuumaps/core";

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
import {
  ActiveColorConfigValue,
  ColorConfigSourceToggleGroup,
  ColorConfigWidget,
} from "@/components/widgets/ColorConfigWidget";
import { useColorConfigWidget } from "@/components/widgets/ColorConfigWidget/hooks";
import {
  ActiveOpacityConfigValue,
  OpacityConfigSourceToggleGroup,
  OpacityConfigWidget,
} from "@/components/widgets/OpacityConfigWidget";
import { useOpacityConfigWidget } from "@/components/widgets/OpacityConfigWidget/hooks";
import { TransformSettingsWidget } from "@/components/widgets/TransformSettingsWidget";
import {
  ActiveVisibilityConfigValue,
  VisibilityConfigSourceToggleGroup,
  VisibilityConfigWidget,
} from "@/components/widgets/VisibilityConfigWidget";
import { useVisibilityConfigWidget } from "@/components/widgets/VisibilityConfigWidget/hooks";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import { LabelsSettingsCategory } from "./category";

export type LabelsAdvancedSettingsWidgetProps = {
  labels: Labels;
  className?: string;
};

export function LabelsAdvancedSettingsWidget({
  labels,
  className,
}: LabelsAdvancedSettingsWidgetProps) {
  const updateLabels = useProjectStore((state) => state.updateLabels);

  const labelColorConfigWidgetAdapter = useColorConfigWidget(
    labels.labelColor,
    (newColorConfig) => updateLabels(labels.id, { labelColor: newColorConfig }),
    defaultLabelColor,
    labels.dataSource.table ?? null,
  );
  const labelVisibilityConfigWidgetAdapter = useVisibilityConfigWidget(
    labels.labelVisibility,
    (newVisibilityConfig) =>
      updateLabels(labels.id, { labelVisibility: newVisibilityConfig }),
    defaultLabelVisibility,
    labels.dataSource.table ?? null,
  );
  const labelOpacityConfigWidgetAdapter = useOpacityConfigWidget(
    labels.labelOpacity,
    (newOpacityConfig) =>
      updateLabels(labels.id, { labelOpacity: newOpacityConfig }),
    defaultLabelOpacity,
    labels.dataSource.table ?? null,
  );

  return (
    <Fieldset className={cn("flex flex-col gap-y-2", className)}>
      <FieldsetLegend className="sr-only">Advanced settings</FieldsetLegend>
      <LayerField
        value={labels.layer}
        onValueChange={(layer) => updateLabels(labels.id, { layer })}
      />
      <Accordion>
        {/* Transform */}
        <AccordionItem value={LabelsSettingsCategory.transform}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Transform</AccordionTrigger>
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <TransformSettingsWidget
              transform={labels.transform}
              onTransformChange={(transform) =>
                updateLabels(labels.id, { transform })
              }
            />
          </AccordionPanel>
        </AccordionItem>
        {/* Label color */}
        <AccordionItem value={LabelsSettingsCategory.labelColor}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Label color</AccordionTrigger>
            <ActiveColorConfigValue
              adapter={labelColorConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel>
            <ColorConfigSourceToggleGroup
              adapter={labelColorConfigWidgetAdapter}
              className="border rounded"
            />
            <ColorConfigWidget adapter={labelColorConfigWidgetAdapter} />
          </AccordionPanel>
        </AccordionItem>
        {/* Label visibility */}
        <AccordionItem value={LabelsSettingsCategory.labelVisibility}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Label visibility</AccordionTrigger>
            <ActiveVisibilityConfigValue
              adapter={labelVisibilityConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel>
            <VisibilityConfigSourceToggleGroup
              adapter={labelVisibilityConfigWidgetAdapter}
              className="border rounded"
            />
            <VisibilityConfigWidget
              adapter={labelVisibilityConfigWidgetAdapter}
            />
          </AccordionPanel>
        </AccordionItem>
        {/* Label opacity */}
        <AccordionItem value={LabelsSettingsCategory.labelOpacity}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Label opacity</AccordionTrigger>
            <ActiveOpacityConfigValue
              adapter={labelOpacityConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel>
            <OpacityConfigSourceToggleGroup
              adapter={labelOpacityConfigWidgetAdapter}
              className="border rounded"
            />
            <OpacityConfigWidget adapter={labelOpacityConfigWidgetAdapter} />
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
    </Fieldset>
  );
}
