import {
  MathUtils,
  type Shapes,
  defaultShapeFillColor,
  defaultShapeFillOpacity,
  defaultShapeFillVisibility,
  defaultShapeOpacity,
  defaultShapeStrokeColor,
  defaultShapeStrokeOpacity,
  defaultShapeStrokeVisibility,
  defaultShapeVisibility,
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
import { Switch } from "@/components/ui/switch";
import { TransformSettingsWidget } from "@/components/widgets/TransformSettingsWidget";
import {
  ActiveColorConfigValue,
  ColorConfigSourceToggleGroup,
  ColorConfigWidget,
} from "@/components/widgets/config/ColorConfigWidget";
import { useColorConfigWidget } from "@/components/widgets/config/ColorConfigWidget/hooks";
import {
  ActiveOpacityConfigValue,
  OpacityConfigSourceToggleGroup,
  OpacityConfigWidget,
} from "@/components/widgets/config/OpacityConfigWidget";
import { useOpacityConfigWidget } from "@/components/widgets/config/OpacityConfigWidget/hooks";
import {
  ActiveVisibilityConfigValue,
  VisibilityConfigSourceToggleGroup,
  VisibilityConfigWidget,
} from "@/components/widgets/config/VisibilityConfigWidget";
import { useVisibilityConfigWidget } from "@/components/widgets/config/VisibilityConfigWidget/hooks";
import { useControlled } from "@/hooks/useControlled";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import { ShapesSettingsCategory } from "./category";

export type ShapesSettingsWidgetProps = {
  shapes: Shapes;
  activeCategory?: ShapesSettingsCategory | null;
  onActiveCategoryChange?: (
    newActiveCategory: ShapesSettingsCategory | null,
  ) => void;
  className?: string;
};

export function ShapesSettingsWidget({
  shapes,
  activeCategory: controlledActiveCategory,
  onActiveCategoryChange: setControlledActiveCategory,
  className,
}: ShapesSettingsWidgetProps) {
  const [activeCategory, setActiveCategory] = useControlled(
    controlledActiveCategory,
    setControlledActiveCategory,
    null,
  );

  const updateShapes = useProjectStore((state) => state.updateShapes);

  const shapeVisibilityConfigWidgetAdapter = useVisibilityConfigWidget(
    shapes.shapeVisibility,
    (newVisibilityConfig) =>
      updateShapes(shapes.id, { shapeVisibility: newVisibilityConfig }),
    defaultShapeVisibility,
    shapes.dataSource.table ?? null,
  );
  const shapeOpacityConfigWidgetAdapter = useOpacityConfigWidget(
    shapes.shapeOpacity,
    (newOpacityConfig) =>
      updateShapes(shapes.id, { shapeOpacity: newOpacityConfig }),
    defaultShapeOpacity,
    shapes.dataSource.table ?? null,
  );
  const shapeFillColorConfigWidgetAdapter = useColorConfigWidget(
    shapes.shapeFillColor,
    (newColorConfig) =>
      updateShapes(shapes.id, { shapeFillColor: newColorConfig }),
    defaultShapeFillColor,
    shapes.dataSource.table ?? null,
  );
  const shapeFillVisibilityConfigWidgetAdapter = useVisibilityConfigWidget(
    shapes.shapeFillVisibility,
    (newVisibilityConfig) =>
      updateShapes(shapes.id, { shapeFillVisibility: newVisibilityConfig }),
    defaultShapeFillVisibility,
    shapes.dataSource.table ?? null,
  );
  const shapeFillOpacityConfigWidgetAdapter = useOpacityConfigWidget(
    shapes.shapeFillOpacity,
    (newOpacityConfig) =>
      updateShapes(shapes.id, { shapeFillOpacity: newOpacityConfig }),
    defaultShapeFillOpacity,
    shapes.dataSource.table ?? null,
  );
  const shapeStrokeColorConfigWidgetAdapter = useColorConfigWidget(
    shapes.shapeStrokeColor,
    (newColorConfig) =>
      updateShapes(shapes.id, { shapeStrokeColor: newColorConfig }),
    defaultShapeStrokeColor,
    shapes.dataSource.table ?? null,
  );
  const shapeStrokeVisibilityConfigWidgetAdapter = useVisibilityConfigWidget(
    shapes.shapeStrokeVisibility,
    (newVisibilityConfig) =>
      updateShapes(shapes.id, {
        shapeStrokeVisibility: newVisibilityConfig,
      }),
    defaultShapeStrokeVisibility,
    shapes.dataSource.table ?? null,
  );
  const shapeStrokeOpacityConfigWidgetAdapter = useOpacityConfigWidget(
    shapes.shapeStrokeOpacity,
    (newOpacityConfig) =>
      updateShapes(shapes.id, { shapeStrokeOpacity: newOpacityConfig }),
    defaultShapeStrokeOpacity,
    shapes.dataSource.table ?? null,
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
            value.length > 0 ? (value[0] as ShapesSettingsCategory) : null,
          )
        }
      >
        {/* General */}
        <AccordionItem value={ShapesSettingsCategory.general}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>General</AccordionTrigger>
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <GeneralShapesSettingsWidget shapes={shapes} />
          </AccordionPanel>
        </AccordionItem>
        {/* Transform */}
        <AccordionItem value={ShapesSettingsCategory.transform}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Transform</AccordionTrigger>
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <TransformSettingsWidget
              transform={shapes.transform}
              onTransformChange={(transform) =>
                updateShapes(shapes.id, { transform })
              }
            />
          </AccordionPanel>
        </AccordionItem>
        {/* Shape visibility */}
        <AccordionItem value={ShapesSettingsCategory.shapeVisibility}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Visibility</AccordionTrigger>
            <ActiveVisibilityConfigValue
              adapter={shapeVisibilityConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <VisibilityConfigSourceToggleGroup
              adapter={shapeVisibilityConfigWidgetAdapter}
              className="border rounded"
            />
            <VisibilityConfigWidget
              adapter={shapeVisibilityConfigWidgetAdapter}
            />
          </AccordionPanel>
        </AccordionItem>
        {/* Shape opacity */}
        <AccordionItem value={ShapesSettingsCategory.shapeOpacity}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Opacity</AccordionTrigger>
            <ActiveOpacityConfigValue
              adapter={shapeOpacityConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <OpacityConfigSourceToggleGroup
              adapter={shapeOpacityConfigWidgetAdapter}
              className="border rounded"
            />
            <OpacityConfigWidget adapter={shapeOpacityConfigWidgetAdapter} />
          </AccordionPanel>
        </AccordionItem>
        {/* Shape fill color */}
        <AccordionItem value={ShapesSettingsCategory.shapeFillColor}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Fill color</AccordionTrigger>
            <ActiveColorConfigValue
              adapter={shapeFillColorConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <ColorConfigSourceToggleGroup
              adapter={shapeFillColorConfigWidgetAdapter}
              className="border rounded"
            />
            <ColorConfigWidget adapter={shapeFillColorConfigWidgetAdapter} />
          </AccordionPanel>
        </AccordionItem>
        {/* Shape fill visibility */}
        <AccordionItem value={ShapesSettingsCategory.shapeFillVisibility}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Fill visibility</AccordionTrigger>
            <ActiveVisibilityConfigValue
              adapter={shapeFillVisibilityConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <VisibilityConfigSourceToggleGroup
              adapter={shapeFillVisibilityConfigWidgetAdapter}
              className="border rounded"
            />
            <VisibilityConfigWidget
              adapter={shapeFillVisibilityConfigWidgetAdapter}
            />
          </AccordionPanel>
        </AccordionItem>
        {/* Shape fill opacity */}
        <AccordionItem value={ShapesSettingsCategory.shapeFillOpacity}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Fill opacity</AccordionTrigger>
            <ActiveOpacityConfigValue
              adapter={shapeFillOpacityConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <OpacityConfigSourceToggleGroup
              adapter={shapeFillOpacityConfigWidgetAdapter}
              className="border rounded"
            />
            <OpacityConfigWidget
              adapter={shapeFillOpacityConfigWidgetAdapter}
            />
          </AccordionPanel>
        </AccordionItem>
        {/* Shape stroke color */}
        <AccordionItem value={ShapesSettingsCategory.shapeStrokeColor}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Stroke color</AccordionTrigger>
            <ActiveColorConfigValue
              adapter={shapeStrokeColorConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <ColorConfigSourceToggleGroup
              adapter={shapeStrokeColorConfigWidgetAdapter}
              className="border rounded"
            />
            <ColorConfigWidget adapter={shapeStrokeColorConfigWidgetAdapter} />
          </AccordionPanel>
        </AccordionItem>
        {/* Shape stroke visibility */}
        <AccordionItem value={ShapesSettingsCategory.shapeStrokeVisibility}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Stroke visibility</AccordionTrigger>
            <ActiveVisibilityConfigValue
              adapter={shapeStrokeVisibilityConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <VisibilityConfigSourceToggleGroup
              adapter={shapeStrokeVisibilityConfigWidgetAdapter}
              className="border rounded"
            />
            <VisibilityConfigWidget
              adapter={shapeStrokeVisibilityConfigWidgetAdapter}
            />
          </AccordionPanel>
        </AccordionItem>
        {/* Shape stroke opacity */}
        <AccordionItem value={ShapesSettingsCategory.shapeStrokeOpacity}>
          <AccordionHeader>
            <AccordionTriggerRightDownIcon />
            <AccordionTrigger>Stroke opacity</AccordionTrigger>
            <ActiveOpacityConfigValue
              adapter={shapeStrokeOpacityConfigWidgetAdapter}
              className="ml-auto text-sm text-slate-600 dark:text-slate-400"
            />
          </AccordionHeader>
          <AccordionPanel className="flex flex-col p-2 pl-6 pb-4 gap-2">
            <OpacityConfigSourceToggleGroup
              adapter={shapeStrokeOpacityConfigWidgetAdapter}
              className="border rounded"
            />
            <OpacityConfigWidget
              adapter={shapeStrokeOpacityConfigWidgetAdapter}
            />
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
    </Fieldset>
  );
}

type GeneralShapesSettingsWidgetProps = {
  shapes: Shapes;
  className?: string;
};

function GeneralShapesSettingsWidget({
  shapes,
  className,
}: GeneralShapesSettingsWidgetProps) {
  const layers = useProjectStore((state) => state.layers);
  const updateShapes = useProjectStore((state) => state.updateShapes);

  return (
    <div className={className}>
      <Field>
        <FieldLabel>Name</FieldLabel>
        <Input
          value={shapes.name}
          onChange={(event) =>
            updateShapes(shapes.id, { name: event.target.value })
          }
        />
      </Field>
      <Field>
        <FieldLabel>Layer</FieldLabel>
        {typeof shapes.layer === "string" ? (
          <SimpleSelect
            items={layers}
            itemLabel={(l) => l.name}
            itemValue={(l) => l.id}
            value={shapes.layer}
            onValueChange={(value) => {
              if (value !== null) {
                updateShapes(shapes.id, { layer: value });
              }
            }}
          />
        ) : (
          <Input disabled value={`column: ${shapes.layer.column}`} readOnly />
        )}
      </Field>
      <Field>
        <FieldLabel>Visibility</FieldLabel>
        <div className="flex flex-row items-center gap-x-2">
          <Switch
            checked={shapes.visibility}
            onCheckedChange={(checked) =>
              updateShapes(shapes.id, { visibility: checked })
            }
          />
          {shapes.visibility ? "Visible" : "Hidden"}
        </div>
      </Field>
      <Field>
        <FieldLabel>Opacity</FieldLabel>
        <Input
          type="number"
          inputMode="decimal"
          step={0.05}
          min={0}
          max={1}
          value={shapes.opacity}
          onChange={(event) => {
            const newValue = event.target.valueAsNumber;
            if (!isNaN(newValue)) {
              updateShapes(shapes.id, {
                opacity: MathUtils.clamp(newValue, 0, 1),
              });
            }
          }}
        />
      </Field>
    </div>
  );
}
