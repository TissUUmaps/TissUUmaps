import type { Layer } from "@tissuumaps/core";

import {
  Collapsible,
  CollapsiblePanel,
  CollapsibleTrigger,
  CollapsibleTriggerRightDownIcon,
} from "@/components/common/collapsible";
import { Field, FieldLabel } from "@/components/common/field";
import { Input } from "@/components/ui/input";
import { TransformSettingsWidget } from "@/components/widgets/TransformSettingsWidget";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

export type LayerSettingsWidgetProps = {
  layer: Layer;
  className?: string;
};

export function LayerSettingsWidget({
  layer,
  className,
}: LayerSettingsWidgetProps) {
  const updateLayer = useProjectStore((state) => state.updateLayer);

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <Field>
        <FieldLabel>Point size factor</FieldLabel>
        <Input
          type="number"
          inputMode="decimal"
          step={0.1}
          min={0}
          value={layer.pointSizeFactor}
          onChange={(event) => {
            const newValue = event.target.valueAsNumber;
            if (!isNaN(newValue)) {
              updateLayer(layer.id, {
                pointSizeFactor: Math.max(0, newValue),
              });
            }
          }}
        />
      </Field>
      <Collapsible>
        <div className="flex flex-row items-center">
          <CollapsibleTriggerRightDownIcon />
          <CollapsibleTrigger>Transform</CollapsibleTrigger>
        </div>
        <CollapsiblePanel className="flex flex-col gap-2 p-2 pb-4 pl-6">
          <TransformSettingsWidget
            transform={layer.transform}
            onTransformChange={(transform) =>
              updateLayer(layer.id, { transform })
            }
          />
        </CollapsiblePanel>
      </Collapsible>
    </div>
  );
}
