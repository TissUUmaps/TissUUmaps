import { RotateCcwIcon, SquareIcon } from "lucide-react";

import { ColorUtils, projectDefaults } from "@tissuumaps/core";

import {
  Collapsible,
  CollapsiblePanel,
  CollapsibleTrigger,
  CollapsibleTriggerRightDownIcon,
} from "@/components/common/collapsible";
import {
  Field,
  FieldControl,
  FieldDescription,
  FieldLabel,
} from "@/components/common/field";
import { SimpleColorPicker } from "@/components/common/simple-color-picker";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

export type RenderSettingsWidgetProps = {
  className?: string;
};

export function RenderSettingsWidget({ className }: RenderSettingsWidgetProps) {
  const glOptions = useProjectStore((state) => state.glOptions);
  const setGLOptions = useProjectStore((state) => state.setGLOptions);
  const viewerBackgroundColor = useProjectStore(
    (state) => state.viewerBackgroundColor,
  );
  const setViewerBackgroundColor = useProjectStore(
    (state) => state.setViewerBackgroundColor,
  );
  const osOptions = useProjectStore((state) => state.osOptions);
  const setOSOptions = useProjectStore((state) => state.setOSOptions);

  const { globalPointSizeFactor } = glOptions.pointsRenderOptions;
  const viewerBackgroundColorHex = ColorUtils.toHex(viewerBackgroundColor);
  const navigatorVisible = osOptions.viewerOptions.showNavigator !== false;

  return (
    <div className={cn("flex flex-col gap-2 pl-6", className)}>
      <Field className="flex flex-col items-start">
        <FieldLabel>Viewer background color</FieldLabel>
        <SimpleColorPicker
          color={viewerBackgroundColor}
          onColorChange={setViewerBackgroundColor}
        >
          <span className="sr-only">Viewer background color</span>
          <SquareIcon fill={viewerBackgroundColorHex} />
          {viewerBackgroundColorHex}
        </SimpleColorPicker>
      </Field>
      <Field>
        <FieldLabel>Navigator</FieldLabel>
        <div className="flex flex-row items-center gap-x-2">
          <Switch
            checked={navigatorVisible}
            onCheckedChange={(checked) =>
              setOSOptions({
                ...osOptions,
                viewerOptions: {
                  ...osOptions.viewerOptions,
                  showNavigator: checked,
                },
              })
            }
          />
          {navigatorVisible ? "Shown" : "Hidden"}
        </div>
      </Field>
      <Field>
        <FieldLabel>Point size factor</FieldLabel>
        <FieldControl
          render={
            <Input
              type="number"
              inputMode="decimal"
              step={0.1}
              min={0}
              value={globalPointSizeFactor}
              onChange={(event) => {
                const newValue = event.target.valueAsNumber;
                if (!isNaN(newValue)) {
                  setGLOptions({
                    ...glOptions,
                    pointsRenderOptions: {
                      ...glOptions.pointsRenderOptions,
                      globalPointSizeFactor: Math.max(0, newValue),
                    },
                  });
                }
              }}
            />
          }
        />
      </Field>
      <Field>
        <FieldLabel>Shape stroke width</FieldLabel>
        <FieldControl
          render={
            <Input
              type="number"
              min={0}
              value={glOptions.shapesRenderOptions.strokeWidth}
              onChange={(event) => {
                const newValue = event.target.valueAsNumber;
                if (!isNaN(newValue)) {
                  setGLOptions({
                    ...glOptions,
                    shapesRenderOptions: {
                      ...glOptions.shapesRenderOptions,
                      strokeWidth: Math.max(0, Math.trunc(newValue)),
                    },
                  });
                }
              }}
            />
          }
        />
        <FieldDescription className="text-muted-foreground text-xs">
          In world coordinates
        </FieldDescription>
      </Field>
      <Collapsible>
        <div className="flex flex-row items-center">
          <CollapsibleTriggerRightDownIcon />
          <CollapsibleTrigger>Advanced rendering</CollapsibleTrigger>
        </div>
        <CollapsiblePanel className="flex flex-col gap-2 p-2 pb-4 pl-6">
          <Field>
            <FieldLabel>Shape edges per scanline</FieldLabel>
            <FieldControl
              render={
                <Input
                  type="number"
                  inputMode="decimal"
                  step={1}
                  min={1}
                  value={glOptions.shapesRenderOptions.edgesPerScanline}
                  onChange={(event) => {
                    const newValue = event.target.valueAsNumber;
                    if (!isNaN(newValue)) {
                      setGLOptions({
                        ...glOptions,
                        shapesRenderOptions: {
                          ...glOptions.shapesRenderOptions,
                          edgesPerScanline: Math.max(1, newValue),
                        },
                      });
                    }
                  }}
                />
              }
            />
            <FieldDescription className="text-muted-foreground text-xs">
              Fewer is faster, but uses more memory
            </FieldDescription>
          </Field>
          <Field>
            <FieldLabel>Shape bin width factor</FieldLabel>
            <FieldControl
              render={
                <Input
                  type="number"
                  inputMode="decimal"
                  step={0.1}
                  min={0.1}
                  value={glOptions.shapesRenderOptions.binWidthFactor}
                  onChange={(event) => {
                    const newValue = event.target.valueAsNumber;
                    if (!isNaN(newValue)) {
                      setGLOptions({
                        ...glOptions,
                        shapesRenderOptions: {
                          ...glOptions.shapesRenderOptions,
                          binWidthFactor: Math.max(0.1, newValue),
                        },
                      });
                    }
                  }}
                />
              }
            />
            <FieldDescription className="text-muted-foreground text-xs">
              Relative to the median shape width; smaller is faster, but uses
              more memory
            </FieldDescription>
          </Field>
          <Field>
            <FieldLabel>Shape padding</FieldLabel>
            <FieldControl
              render={
                <Input
                  type="number"
                  inputMode="decimal"
                  step={0.1}
                  min={0}
                  value={glOptions.shapesRenderOptions.shapePadding}
                  onChange={(event) => {
                    const newValue = event.target.valueAsNumber;
                    if (!isNaN(newValue)) {
                      setGLOptions({
                        ...glOptions,
                        shapesRenderOptions: {
                          ...glOptions.shapesRenderOptions,
                          shapePadding: Math.max(0, newValue),
                        },
                      });
                    }
                  }}
                />
              }
            />
            <FieldDescription className="text-muted-foreground text-xs">
              Relative to the median shape size; larger lets strokes reach
              further, but is slower
            </FieldDescription>
          </Field>
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground self-start"
            onClick={() => {
              const { edgesPerScanline, binWidthFactor, shapePadding } =
                projectDefaults.glOptions.shapesRenderOptions;
              setGLOptions({
                ...glOptions,
                shapesRenderOptions: {
                  ...glOptions.shapesRenderOptions,
                  edgesPerScanline,
                  binWidthFactor,
                  shapePadding,
                },
              });
            }}
          >
            <RotateCcwIcon />
            Reset to defaults
          </Button>
        </CollapsiblePanel>
      </Collapsible>
    </div>
  );
}
