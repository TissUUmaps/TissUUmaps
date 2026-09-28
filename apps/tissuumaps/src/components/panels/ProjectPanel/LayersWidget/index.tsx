import { PlusIcon } from "lucide-react";
import { useMemo } from "react";

import { type Layer, type TableColumnRef, createLayer } from "@tissuumaps/core";

import { OpacityControl } from "@/components/common/opacity-control";
import { VisibilityButton } from "@/components/common/visibility-button";
import {
  type ObjectKind,
  objectKindIcons,
} from "@/components/object-kind-icons";
import { Button } from "@/components/ui/button";
import {
  SortableObjectItem,
  SortableObjectList,
} from "@/components/widgets/ObjectList";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import { LayerSettingsWidget } from "./LayerSettingsWidget";

export type LayersWidgetProps = {
  className?: string;
};

export function LayersWidget({ className }: LayersWidgetProps) {
  const layers = useProjectStore((state) => state.layers);
  const addLayer = useProjectStore((state) => state.addLayer);
  const moveLayer = useProjectStore((state) => state.moveLayer);

  const objectsByLayer = useObjectsByLayer();

  return (
    <div className={cn("flex flex-col gap-y-2", className)}>
      <SortableObjectList objects={layers} onMove={moveLayer}>
        {(layer, index) => (
          <LayerAccordionItem
            key={layer.id}
            layer={layer}
            index={index}
            objects={objectsByLayer.get(layer.id) ?? []}
          />
        )}
      </SortableObjectList>
      <Button
        variant="outline"
        className="w-full"
        onClick={() => {
          const layer = createLayer({
            id: crypto.randomUUID(),
            name: `Layer ${layers.length + 1}`,
          });
          addLayer(layer);
        }}
      >
        <PlusIcon className="size-4" />
        Add
      </Button>
    </div>
  );
}

type LayerObject = {
  id: string;
  name: string;
  kind: ObjectKind;
};

function useObjectsByLayer(): Map<string, LayerObject[]> {
  const images = useProjectStore((state) => state.images);
  const labels = useProjectStore((state) => state.labels);
  const points = useProjectStore((state) => state.points);
  const shapes = useProjectStore((state) => state.shapes);

  return useMemo(() => {
    const objectsByLayer = new Map<string, LayerObject[]>();
    const add = (layer: string | TableColumnRef, object: LayerObject) => {
      if (typeof layer === "string") {
        const objects = objectsByLayer.get(layer);
        if (objects === undefined) {
          objectsByLayer.set(layer, [object]);
        } else {
          objects.push(object);
        }
      }
    };
    for (const image of images) {
      add(image.layer, { id: image.id, name: image.name, kind: "image" });
    }
    for (const l of labels) {
      add(l.layer, { id: l.id, name: l.name, kind: "labels" });
    }
    for (const p of points) {
      add(p.layer, { id: p.id, name: p.name, kind: "points" });
    }
    for (const s of shapes) {
      add(s.layer, { id: s.id, name: s.name, kind: "shapes" });
    }
    return objectsByLayer;
  }, [images, labels, points, shapes]);
}

type LayerAccordionItemProps = {
  layer: Layer;
  index: number;
  objects: LayerObject[];
};

function LayerAccordionItem({
  layer,
  index,
  objects,
}: LayerAccordionItemProps) {
  const updateLayer = useProjectStore((state) => state.updateLayer);
  const deleteLayer = useProjectStore((state) => state.deleteLayer);

  return (
    <SortableObjectItem
      id={layer.id}
      index={index}
      name={layer.name}
      objectLabel="layer"
      onRename={(name) => updateLayer(layer.id, { name })}
      dimmed={!layer.visibility}
      leadingControls={
        <VisibilityButton
          visible={layer.visibility}
          onVisibleChange={(visibility) =>
            updateLayer(layer.id, { visibility })
          }
          objectLabel="layer"
          name={layer.name}
        />
      }
      trailingControls={
        <>
          {objects.length > 0 && (
            <span className="bg-muted text-muted-foreground mr-1 rounded-full px-1.5 text-xs tabular-nums">
              {objects.length}
            </span>
          )}
          <OpacityControl
            opacity={layer.opacity}
            name={layer.name}
            onOpacityChange={(opacity) => updateLayer(layer.id, { opacity })}
          />
        </>
      }
      deleteDisabledReason={
        objects.length > 0
          ? "Cannot delete a layer that contains objects"
          : undefined
      }
      onDelete={() => deleteLayer(layer.id)}
    >
      {objects.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {objects.map((object) => {
            const Icon = objectKindIcons[object.kind];
            return (
              <span
                key={object.id}
                className="bg-background inline-flex h-6 max-w-full items-center gap-1 rounded-full border px-2 text-xs"
              >
                <Icon className="text-muted-foreground size-3.5 shrink-0" />
                <span className="truncate">{object.name}</span>
              </span>
            );
          })}
        </div>
      )}
      <LayerSettingsWidget layer={layer} />
    </SortableObjectItem>
  );
}
