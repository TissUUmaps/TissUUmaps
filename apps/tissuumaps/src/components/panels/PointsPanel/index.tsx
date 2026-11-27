import { type Points, ProjectUtils } from "@tissuumaps/core";

import { OpacityControl } from "@/components/common/opacity-control";
import { VisibilityButton } from "@/components/common/visibility-button";
import { AddDataObjectButton } from "@/components/controls/AddDataObjectButton";
import {
  SortableObjectList,
  SortableObjectListItem,
} from "@/components/controls/ObjectList";
import { useAddPointsDialogParams } from "@/components/dialogs/AddDataObjectDialog/hooks";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { DataSourceWidget } from "@/components/widgets/DataSourceWidget";
import { pointsDataCache } from "@/data/cache";
import { usePointsData } from "@/data/hooks/useData";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useDataStore } from "@/stores/data";
import { useProjectStore } from "@/stores/project";

import { PointsAnnotationsWidget } from "./PointsAnnotationsWidget";
import { PointsSettingsWidget } from "./PointsSettingsWidget";

export type PointsPanelProps = {
  className?: string;
};

export function PointsPanel({ className }: PointsPanelProps) {
  const expandedPointsIds = useAppStore((state) => state.expandedPointsIds);
  const setExpandedPointsIds = useAppStore(
    (state) => state.setExpandedPointsIds,
  );
  const addDialogParams = useAddPointsDialogParams();

  const points = useProjectStore((state) => state.points);
  const movePoints = useProjectStore((state) => state.movePoints);

  return (
    <div className={cn("flex flex-col gap-y-2", className)}>
      <SortableObjectList
        objects={points}
        onMove={movePoints}
        expandedIds={expandedPointsIds}
        onExpandedIdsChange={setExpandedPointsIds}
      >
        {(currentPoints, index) => (
          <PointsAccordionItem
            key={currentPoints.id}
            points={currentPoints}
            index={index}
          />
        )}
      </SortableObjectList>
      <AddDataObjectButton {...addDialogParams} />
    </div>
  );
}

type PointsAccordionItemProps = {
  points: Points;
  index: number;
};

function PointsAccordionItem({ points, index }: PointsAccordionItemProps) {
  const pointsDataProviders = useAppStore((state) => state.pointsDataProviders);

  const updatePoints = useProjectStore((state) => state.updatePoints);
  const deletePoints = useProjectStore((state) => state.deletePoints);
  const loaded = useDataStore((state) => state.pointsDataRefs.has(points.id));
  const visible = useProjectStore((state) =>
    ProjectUtils.mayBeVisible(points, state.layers),
  );

  const pointsData = usePointsData(points.id);

  return (
    <SortableObjectListItem
      id={points.id}
      index={index}
      name={points.name}
      objectLabel="point cloud"
      onRename={(name) => updatePoints(points.id, { name })}
      dimmed={!points.visibility}
      leadingControls={
        <VisibilityButton
          visible={points.visibility}
          onVisibleChange={(visibility) =>
            updatePoints(points.id, { visibility })
          }
          objectLabel="point cloud"
          name={points.name}
        />
      }
      trailingControls={
        <>
          <InputGroup className="h-6 w-16 shrink-0">
            <InputGroupInput
              type="number"
              inputMode="decimal"
              step={0.1}
              min={0}
              aria-label={`Point size factor of ${points.name}`}
              title="Point size factor"
              value={points.pointSizeFactor}
              onChange={(event) => {
                const newValue = event.target.valueAsNumber;
                if (!isNaN(newValue)) {
                  updatePoints(points.id, {
                    pointSizeFactor: Math.max(0, newValue),
                  });
                }
              }}
              className="h-6 px-1.5 text-xs tabular-nums"
            />
            <InputGroupAddon align="inline-end" className="pr-1.5 text-xs">
              ×
            </InputGroupAddon>
          </InputGroup>
          <OpacityControl
            opacity={points.opacity}
            name={points.name}
            onOpacityChange={(opacity) => updatePoints(points.id, { opacity })}
          />
        </>
      }
      unloadDisabled={!loaded}
      onUnload={() => {
        updatePoints(points.id, { visibility: false });
        pointsDataCache.release(points.id);
      }}
      onDelete={() => deletePoints(points.id)}
    >
      <DataSourceWidget
        dataSource={points.dataSource}
        dataProviders={pointsDataProviders}
        onDataSourceChange={(newDataSource) => {
          updatePoints(points.id, { dataSource: newDataSource });
        }}
        className="bg-card"
      />
      <PointsSettingsWidget points={points} className="bg-card" />
      {pointsData !== null && (
        <PointsAnnotationsWidget
          points={points}
          data={pointsData}
          className="bg-card"
        />
      )}
      {!loaded && !visible && (
        <p className="text-muted-foreground text-sm">
          Show the points and their layer to load their annotations.
        </p>
      )}
    </SortableObjectListItem>
  );
}
