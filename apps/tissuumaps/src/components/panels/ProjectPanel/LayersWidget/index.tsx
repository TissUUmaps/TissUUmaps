import { DragDropProvider } from "@dnd-kit/react";
import { useSortable } from "@dnd-kit/react/sortable";
import {
  EyeIcon,
  EyeOffIcon,
  GripVertical,
  PlusIcon,
  Trash2Icon,
} from "lucide-react";
import { useMemo } from "react";

import { type Layer, MathUtils, createLayer } from "@tissuumaps/core";

import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  AccordionTrigger,
  AccordionTriggerDownUpIcon,
} from "@/components/common/accordion";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { Button } from "@/components/ui/button";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { useTopFirstSortable } from "@/hooks/useTopFirstSortable";
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
  const { topFirstItems, onDragEnd } = useTopFirstSortable(layers, moveLayer);

  return (
    <div className={cn("flex flex-col gap-y-2", className)}>
      <DragDropProvider onDragEnd={onDragEnd}>
        <Accordion multiple className="gap-y-2">
          {topFirstItems.map((layer, index) => (
            <LayerAccordionItem key={layer.id} layer={layer} index={index} />
          ))}
        </Accordion>
      </DragDropProvider>
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

function useLayerObjects(layerId: string) {
  const images = useProjectStore((state) => state.images);
  const labels = useProjectStore((state) => state.labels);
  const points = useProjectStore((state) => state.points);
  const shapes = useProjectStore((state) => state.shapes);

  return useMemo(() => {
    const names: string[] = [];
    for (const image of images) {
      if (image.layer === layerId) names.push(image.name);
    }
    for (const l of labels) {
      if (l.layer === layerId) names.push(l.name);
    }
    for (const p of points) {
      if (p.layer === layerId) names.push(p.name);
    }
    for (const s of shapes) {
      if (s.layer === layerId) names.push(s.name);
    }
    return names;
  }, [layerId, images, labels, points, shapes]);
}

type LayerAccordionItemProps = {
  layer: Layer;
  index: number;
};

function LayerAccordionItem({ layer, index }: LayerAccordionItemProps) {
  const updateLayer = useProjectStore((state) => state.updateLayer);
  const deleteLayer = useProjectStore((state) => state.deleteLayer);
  const confirm = useConfirmDialog();

  const objectNames = useLayerObjects(layer.id);
  const hasObjects = objectNames.length > 0;

  const { ref, handleRef } = useSortable({ id: layer.id, index });

  return (
    <div ref={ref}>
      <AccordionItem
        value={layer.id}
        className="border rounded-md bg-sidebar p-2"
      >
        <AccordionHeader>
          <GripVertical ref={handleRef} />
          <div className="flex-1 w-full">
            <AccordionTrigger className="w-full cursor-pointer">
              {layer.name}
              {hasObjects && (
                <span className="ml-1 text-xs text-muted-foreground">
                  ({objectNames.length})
                </span>
              )}
            </AccordionTrigger>
          </div>
          <div className="ml-auto flex flex-row items-center gap-x-2">
            <InputGroup className="w-20">
              <InputGroupAddon>&alpha;</InputGroupAddon>
              <InputGroupInput
                type="number"
                aria-label="Opacity"
                inputMode="decimal"
                step={0.05}
                min={0}
                max={1}
                value={layer.opacity}
                onChange={(event) => {
                  const newValue = event.target.valueAsNumber;
                  if (!isNaN(newValue)) {
                    updateLayer(layer.id, {
                      opacity: MathUtils.clamp(newValue, 0, 1),
                    });
                  }
                }}
              />
            </InputGroup>
            <Button
              variant="ghost"
              aria-label={layer.visibility ? "Hide layer" : "Show layer"}
              onClick={() =>
                updateLayer(layer.id, { visibility: !layer.visibility })
              }
            >
              {layer.visibility ? <EyeIcon /> : <EyeOffIcon />}
            </Button>
            <Button
              variant="ghost"
              disabled={hasObjects}
              aria-label="Delete layer"
              onClick={() => {
                void confirm({
                  title: "Delete layer",
                  body: "Are you sure you want to delete this layer? This action cannot be undone.",
                }).then((confirmed) => {
                  if (confirmed) {
                    deleteLayer(layer.id);
                  }
                });
              }}
              title={
                hasObjects
                  ? "Cannot delete a layer that contains objects"
                  : "Delete layer"
              }
            >
              <Trash2Icon />
            </Button>
          </div>
          <AccordionTriggerDownUpIcon />
        </AccordionHeader>
        <AccordionPanel className="pt-2 flex flex-col gap-y-2">
          {hasObjects && (
            <div className="text-xs text-muted-foreground px-1">
              {objectNames.join(", ")}
            </div>
          )}
          <LayerSettingsWidget layer={layer} className="bg-card" />
        </AccordionPanel>
      </AccordionItem>
    </div>
  );
}
