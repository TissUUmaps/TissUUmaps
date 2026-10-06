import { type Image, createImage } from "@tissuumaps/core";

import { OpacityControl } from "@/components/common/opacity-control";
import { VisibilityButton } from "@/components/common/visibility-button";
import { AddDataObjectButton } from "@/components/widgets/AddDataObjectButton";
import { DataSourceWidget } from "@/components/widgets/DataSourceWidget";
import {
  SortableObjectList,
  SortableObjectListItem,
} from "@/components/widgets/ObjectList";
import { useImageData } from "@/hooks/useData";
import { useExpandedImageIds } from "@/hooks/useExpandedIds";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import { ChannelSettingsWidget } from "./ChannelSettingsWidget";
import { ImageSettingsWidget } from "./ImageSettingsWidget";

export type ImagesPanelProps = {
  /** Brings the panel to the front */
  onShow: () => void;
  className?: string;
};

export function ImagesPanel({ onShow, className }: ImagesPanelProps) {
  const [expandedIds, setExpandedIds] = useExpandedImageIds(onShow);
  const imageDataProviders = useAppStore((state) => state.imageDataProviders);

  const layers = useProjectStore((state) => state.layers);
  const images = useProjectStore((state) => state.images);
  const addImage = useProjectStore((state) => state.addImage);
  const moveImage = useProjectStore((state) => state.moveImage);

  return (
    <div className={cn("flex flex-col gap-y-2", className)}>
      <SortableObjectList
        objects={images}
        onMove={moveImage}
        expandedIds={expandedIds}
        onExpandedIdsChange={setExpandedIds}
      >
        {(image, index) => (
          <ImageAccordionItem key={image.id} image={image} index={index} />
        )}
      </SortableObjectList>
      <AddDataObjectButton
        title="Add image"
        layers={layers}
        dataProviders={imageDataProviders}
        onAdd={(name, layerId, dataSource) => {
          if (!layerId) return;
          const image = createImage({
            id: crypto.randomUUID(),
            name,
            dataSource,
            layer: layerId,
          });
          addImage(image);
        }}
      />
    </div>
  );
}

type ImageAccordionItemProps = {
  image: Image;
  index: number;
};

function ImageAccordionItem({ image, index }: ImageAccordionItemProps) {
  const imageDataProviders = useAppStore((state) => state.imageDataProviders);

  const updateImage = useProjectStore((state) => state.updateImage);
  const deleteImage = useProjectStore((state) => state.deleteImage);

  const imageData = useImageData(image.id);
  const sizeC = imageData?.getSizeC();

  return (
    <SortableObjectListItem
      id={image.id}
      index={index}
      name={image.name}
      objectLabel="image"
      onRename={(name) => updateImage(image.id, { name })}
      dimmed={!image.visibility}
      leadingControls={
        <VisibilityButton
          visible={image.visibility}
          onVisibleChange={(visibility) =>
            updateImage(image.id, { visibility })
          }
          objectLabel="image"
          name={image.name}
        />
      }
      trailingControls={
        <OpacityControl
          opacity={image.opacity}
          name={image.name}
          onOpacityChange={(opacity) => updateImage(image.id, { opacity })}
        />
      }
      onDelete={() => deleteImage(image.id)}
    >
      <DataSourceWidget
        dataSource={image.dataSource}
        dataProviders={imageDataProviders}
        onDataSourceChange={(newDataSource) => {
          updateImage(image.id, { dataSource: newDataSource });
        }}
        className="bg-card"
      />
      <ImageSettingsWidget image={image} className="bg-card" />
      {imageData !== null && sizeC !== undefined && (
        <ChannelSettingsWidget
          image={image}
          data={imageData}
          sizeC={sizeC}
          className="bg-card"
        />
      )}
    </SortableObjectListItem>
  );
}
