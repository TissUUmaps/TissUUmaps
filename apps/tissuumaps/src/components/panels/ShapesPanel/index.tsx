import { useState } from "react";

import { type Shapes, createShapes } from "@tissuumaps/core";

import { OpacityControl } from "@/components/common/opacity-control";
import { VisibilityButton } from "@/components/common/visibility-button";
import { AddDataObjectButton } from "@/components/widgets/AddDataObjectButton";
import { DataSourceWidget } from "@/components/widgets/DataSourceWidget";
import {
  SortableObjectList,
  SortableObjectListItem,
} from "@/components/widgets/ObjectList";
import { useShapesData } from "@/hooks/useData";
import { useExpandedShapesIds } from "@/hooks/useExpandedIds";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import { ShapesAnnotationsWidget } from "./ShapesAnnotationsWidget";
import { ShapesSettingsWidget } from "./ShapesSettingsWidget";
import type { ShapesSettingsCategory } from "./category";

export type ShapesPanelProps = {
  /** Brings the panel to the front */
  onShow: () => void;
  className?: string;
};

export function ShapesPanel({ onShow, className }: ShapesPanelProps) {
  const [expandedIds, setExpandedIds] = useExpandedShapesIds(onShow);
  const shapesDataProviders = useAppStore((state) => state.shapesDataProviders);

  const layers = useProjectStore((state) => state.layers);
  const shapes = useProjectStore((state) => state.shapes);
  const addShapes = useProjectStore((state) => state.addShapes);
  const moveShapes = useProjectStore((state) => state.moveShapes);

  return (
    <div className={cn("flex flex-col gap-y-2", className)}>
      <SortableObjectList
        objects={shapes}
        onMove={moveShapes}
        expandedIds={expandedIds}
        onExpandedIdsChange={setExpandedIds}
      >
        {(currentShapes, index) => (
          <ShapesAccordionItem
            key={currentShapes.id}
            shapes={currentShapes}
            index={index}
          />
        )}
      </SortableObjectList>
      <AddDataObjectButton
        title="Add shapes"
        layers={layers}
        dataProviders={shapesDataProviders}
        onAdd={(name, layerId, dataSource) => {
          if (!layerId) return;
          const newShapes = createShapes({
            id: crypto.randomUUID(),
            name,
            dataSource,
            layer: layerId,
          });
          addShapes(newShapes);
        }}
      />
    </div>
  );
}

type ShapesAccordionItemProps = {
  shapes: Shapes;
  index: number;
};

function ShapesAccordionItem({ shapes, index }: ShapesAccordionItemProps) {
  const shapesDataProviders = useAppStore((state) => state.shapesDataProviders);

  const updateShapes = useProjectStore((state) => state.updateShapes);
  const deleteShapes = useProjectStore((state) => state.deleteShapes);

  const shapesData = useShapesData(shapes.id);

  const [activeSettingsCategory, setActiveSettingsCategory] =
    useState<ShapesSettingsCategory | null>(null);

  return (
    <SortableObjectListItem
      id={shapes.id}
      index={index}
      name={shapes.name}
      objectLabel="shape cloud"
      onRename={(name) => updateShapes(shapes.id, { name })}
      dimmed={!shapes.visibility}
      leadingControls={
        <VisibilityButton
          visible={shapes.visibility}
          onVisibleChange={(visibility) =>
            updateShapes(shapes.id, { visibility })
          }
          objectLabel="shape cloud"
          name={shapes.name}
        />
      }
      trailingControls={
        <OpacityControl
          opacity={shapes.opacity}
          name={shapes.name}
          onOpacityChange={(opacity) => updateShapes(shapes.id, { opacity })}
        />
      }
      onDelete={() => deleteShapes(shapes.id)}
    >
      <DataSourceWidget
        dataSource={shapes.dataSource}
        dataProviders={shapesDataProviders}
        onDataSourceChange={(newDataSource) => {
          updateShapes(shapes.id, { dataSource: newDataSource });
        }}
        className="bg-card"
      />
      <ShapesSettingsWidget
        shapes={shapes}
        activeCategory={activeSettingsCategory}
        onActiveCategoryChange={setActiveSettingsCategory}
        className="bg-card"
      />
      {shapesData !== null && (
        <ShapesAnnotationsWidget
          shapes={shapes}
          data={shapesData}
          activeSettingsCategory={activeSettingsCategory}
          className="bg-card"
        />
      )}
    </SortableObjectListItem>
  );
}
