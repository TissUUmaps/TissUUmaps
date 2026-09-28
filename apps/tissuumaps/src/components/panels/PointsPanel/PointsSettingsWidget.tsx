import {
  type Points,
  defaultPointColor,
  defaultPointMarker,
  defaultPointOpacity,
  defaultPointSize,
  defaultPointSizeUnit,
  defaultPointVisibility,
} from "@tissuumaps/core";

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
import { Input } from "@/components/ui/input";
import { formatTableColumn } from "@/components/widgets/TableColumnField/formatTableColumn";
import { TransformSettingsWidget } from "@/components/widgets/TransformSettingsWidget";
import {
  ActiveColorConfigValue,
  ColorConfigSourceToggleGroup,
  ColorConfigWidget,
} from "@/components/widgets/config/ColorConfigWidget";
import { useColorConfigWidget } from "@/components/widgets/config/ColorConfigWidget/hooks";
import {
  ActiveMarkerConfigValue,
  MarkerConfigSourceToggleGroup,
  MarkerConfigWidget,
} from "@/components/widgets/config/MarkerConfigWidget";
import { useMarkerConfigWidget } from "@/components/widgets/config/MarkerConfigWidget/hooks";
import {
  ActiveOpacityConfigValue,
  OpacityConfigSourceToggleGroup,
  OpacityConfigWidget,
} from "@/components/widgets/config/OpacityConfigWidget";
import { useOpacityConfigWidget } from "@/components/widgets/config/OpacityConfigWidget/hooks";
import {
  ActiveSizeConfigValue,
  SizeConfigSourceToggleGroup,
  SizeConfigWidget,
} from "@/components/widgets/config/SizeConfigWidget";
import { useSizeConfigWidget } from "@/components/widgets/config/SizeConfigWidget/hooks";
import {
  ActiveVisibilityConfigValue,
  VisibilityConfigSourceToggleGroup,
  VisibilityConfigWidget,
} from "@/components/widgets/config/VisibilityConfigWidget";
import { useVisibilityConfigWidget } from "@/components/widgets/config/VisibilityConfigWidget/hooks";
import { useControlled } from "@/hooks/useControlled";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import { PointsSettingsCategory } from "./category";

export type PointsSettingsWidgetProps = {
  points: Points;
  activeCategory?: PointsSettingsCategory | null;
  onActiveCategoryChange?: (
    newActiveCategory: PointsSettingsCategory | null,
  ) => void;
  className?: string;
};

export function PointsSettingsWidget({
  points,
  activeCategory: controlledActiveCategory,
  onActiveCategoryChange: setControlledActiveCategory,
  className,
}: PointsSettingsWidgetProps) {
  const [activeCategory, setActiveCategory] = useControlled(
    controlledActiveCategory,
    setControlledActiveCategory,
    null,
  );

  const updatePoints = useProjectStore((state) => state.updatePoints);

  const pointMarkerConfigWidgetAdapter = useMarkerConfigWidget(
    points.pointMarker,
    (newMarkerConfig) =>
      updatePoints(points.id, { pointMarker: newMarkerConfig }),
    defaultPointMarker,
    points.dataSource.table ?? null,
  );
  const pointSizeConfigWidgetAdapter = useSizeConfigWidget(
    points.pointSize,
    (newSizeConfig) => updatePoints(points.id, { pointSize: newSizeConfig }),
    defaultPointSize,
    defaultPointSizeUnit,
    points.dataSource.table ?? null,
  );
  const pointColorConfigWidgetAdapter = useColorConfigWidget(
    points.pointColor,
    (newColorConfig) => updatePoints(points.id, { pointColor: newColorConfig }),
    defaultPointColor,
    points.dataSource.table ?? null,
  );
  const pointVisibilityConfigWidgetAdapter = useVisibilityConfigWidget(
    points.pointVisibility,
    (newVisibilityConfig) =>
      updatePoints(points.id, { pointVisibility: newVisibilityConfig }),
    defaultPointVisibility,
    points.dataSource.table ?? null,
  );
  const pointOpacityConfigWidgetAdapter = useOpacityConfigWidget(
    points.pointOpacity,
    (newOpacityConfig) =>
      updatePoints(points.id, { pointOpacity: newOpacityConfig }),
    defaultPointOpacity,
    points.dataSource.table ?? null,
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
            value.length > 0 ? (value[0] as PointsSettingsCategory) : null,
          )
        }
      >
        {/* General */}
        <AccordionItem value={PointsSettingsCategory.general}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>General</AccordionTrigger>
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <GeneralPointsSettingsWidget points={points} />
          </AccordionPanel>
        </AccordionItem>
        {/* Transform */}
        <AccordionItem value={PointsSettingsCategory.transform}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Transform</AccordionTrigger>
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <TransformSettingsWidget
              transform={points.transform}
              onTransformChange={(transform) =>
                updatePoints(points.id, { transform })
              }
            />
          </AccordionPanel>
        </AccordionItem>
        {/* Point marker */}
        <AccordionItem value={PointsSettingsCategory.pointMarker}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Point marker</AccordionTrigger>
            <ActiveMarkerConfigValue
              adapter={pointMarkerConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <MarkerConfigSourceToggleGroup
              adapter={pointMarkerConfigWidgetAdapter}
              className="border rounded"
            />
            <MarkerConfigWidget adapter={pointMarkerConfigWidgetAdapter} />
          </AccordionPanel>
        </AccordionItem>
        {/* Point size */}
        <AccordionItem value={PointsSettingsCategory.pointSize}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Point size</AccordionTrigger>
            <ActiveSizeConfigValue
              adapter={pointSizeConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <SizeConfigSourceToggleGroup
              adapter={pointSizeConfigWidgetAdapter}
              className="border rounded"
            />
            <SizeConfigWidget adapter={pointSizeConfigWidgetAdapter} />
          </AccordionPanel>
        </AccordionItem>
        {/* Point color */}
        <AccordionItem value={PointsSettingsCategory.pointColor}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Point color</AccordionTrigger>
            <ActiveColorConfigValue
              adapter={pointColorConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <ColorConfigSourceToggleGroup
              adapter={pointColorConfigWidgetAdapter}
              className="border rounded"
            />
            <ColorConfigWidget adapter={pointColorConfigWidgetAdapter} />
          </AccordionPanel>
        </AccordionItem>
        {/* Point visibility */}
        <AccordionItem value={PointsSettingsCategory.pointVisibility}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Point visibility</AccordionTrigger>
            <ActiveVisibilityConfigValue
              adapter={pointVisibilityConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <VisibilityConfigSourceToggleGroup
              adapter={pointVisibilityConfigWidgetAdapter}
              className="border rounded"
            />
            <VisibilityConfigWidget
              adapter={pointVisibilityConfigWidgetAdapter}
            />
          </AccordionPanel>
        </AccordionItem>
        {/* Point opacity */}
        <AccordionItem value={PointsSettingsCategory.pointOpacity}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Point opacity</AccordionTrigger>
            <ActiveOpacityConfigValue
              adapter={pointOpacityConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <OpacityConfigSourceToggleGroup
              adapter={pointOpacityConfigWidgetAdapter}
              className="border rounded"
            />
            <OpacityConfigWidget adapter={pointOpacityConfigWidgetAdapter} />
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
    </Fieldset>
  );
}

type GeneralPointsSettingsWidgetProps = {
  points: Points;
  className?: string;
};

function GeneralPointsSettingsWidget({
  points,
  className,
}: GeneralPointsSettingsWidgetProps) {
  const layers = useProjectStore((state) => state.layers);
  const tables = useProjectStore((state) => state.tables);
  const updatePoints = useProjectStore((state) => state.updatePoints);

  return (
    <div className={className}>
      <Field>
        <FieldLabel>Layer</FieldLabel>
        {typeof points.layer === "string" ? (
          <SimpleSelect
            items={layers}
            itemLabel={(l) => l.name}
            itemValue={(l) => l.id}
            value={points.layer}
            onValueChange={(value) => {
              if (value !== null) {
                updatePoints(points.id, { layer: value });
              }
            }}
          />
        ) : (
          <Input
            disabled
            value={`column: ${formatTableColumn(points.layer, tables)}`}
            readOnly
          />
        )}
      </Field>
      <Field>
        <FieldLabel>Point size factor</FieldLabel>
        <Input
          type="number"
          inputMode="decimal"
          step={0.1}
          min={0}
          value={points.pointSizeFactor}
          onChange={(event) => {
            const newValue = event.target.valueAsNumber;
            if (!isNaN(newValue)) {
              updatePoints(points.id, {
                pointSizeFactor: Math.max(0, newValue),
              });
            }
          }}
        />
      </Field>
    </div>
  );
}
