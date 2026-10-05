import { useEffect, useState } from "react";

import {
  type Color,
  ColorUtils,
  type Image,
  type ImageChannel,
  ImageChannelViewMode,
  type ImageData,
  ImageUtils,
} from "@tissuumaps/core";

import {
  Collapsible,
  CollapsiblePanel,
  CollapsibleTrigger,
  CollapsibleTriggerRightDownIcon,
} from "@/components/common/collapsible";
import { Fieldset, FieldsetLegend } from "@/components/common/fieldset";
import { OpacityControl } from "@/components/common/opacity-control";
import { VisibilityButton } from "@/components/common/visibility-button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import ColorPicker from "../../common/color-picker";
import { ContrastRangeWidget } from "./ContrastRangeWidget";
import { channelViewModeLabels } from "./channelViewMode";

export type ChannelSettingsWidgetProps = {
  image: Image;
  data: ImageData;
  sizeC: number;
  className?: string;
};

export function ChannelSettingsWidget({
  image,
  data,
  sizeC,
  className,
}: ChannelSettingsWidgetProps) {
  const updateImage = useProjectStore((state) => state.updateImage);
  const setImageChannelPreview = useAppStore(
    (state) => state.setImageChannelPreview,
  );
  const [expandedChannels, setExpandedChannels] = useState<number[]>([]);

  // End the preview when the widget goes away while a channel is hovered
  useEffect(() => () => setImageChannelPreview(null), [setImageChannelPreview]);

  // Base UI's accordion would take the radios from the radio group, which then
  // no longer moves the active channel on arrow keys
  const rows = (
    <div className="flex flex-col gap-y-1 text-sm">
      {Array.from({ length: sizeC }, (_, c) => (
        <ChannelSettingsRow
          key={c}
          image={image}
          data={data}
          channelIndex={c}
          expanded={expandedChannels.includes(c)}
          onExpandedChange={(expanded) =>
            setExpandedChannels((channels) =>
              expanded ? [...channels, c] : channels.filter((i) => i !== c),
            )
          }
        />
      ))}
    </div>
  );

  return (
    <Fieldset
      className={cn("flex flex-col gap-y-2 border rounded-md p-2", className)}
    >
      <FieldsetLegend className="flex flex-row items-center gap-x-1 font-medium text-foreground">
        Channels
        <ToggleGroup
          size="sm"
          value={[image.channelViewMode]}
          onValueChange={(value) => {
            if (value.length > 0) {
              updateImage(image.id, {
                channelViewMode: value[0] as ImageChannelViewMode,
              });
            }
          }}
          className="ml-auto border rounded"
        >
          {Object.values(ImageChannelViewMode).map((mode) => (
            <ToggleGroupItem
              key={mode}
              value={mode}
              className={
                image.channelViewMode === mode ? "font-medium" : "font-normal"
              }
            >
              {channelViewModeLabels[mode]}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </FieldsetLegend>
      {image.channelViewMode !== ImageChannelViewMode.composite ? (
        <RadioGroup
          value={String(ImageUtils.getActiveChannel(image, sizeC))}
          onValueChange={(value) => {
            if (typeof value === "string") {
              updateImage(image.id, { activeChannel: Number(value) });
            }
          }}
          // Base UI moves the active channel on arrow keys from anything inside
          // the group but text inputs, so leave them to the rows' number inputs
          // and sliders
          onKeyDown={(event) => {
            if (
              (event.target as HTMLElement).getAttribute("role") !== "radio"
            ) {
              event.preventBaseUIHandler();
            }
          }}
        >
          {rows}
        </RadioGroup>
      ) : (
        rows
      )}
    </Fieldset>
  );
}

type ChannelSettingsRowProps = {
  image: Image;
  data: ImageData;
  channelIndex: number;
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
  className?: string;
};

function ChannelSettingsRow({
  image,
  data,
  channelIndex: c,
  expanded,
  onExpandedChange,
  className,
}: ChannelSettingsRowProps) {
  const updateImage = useProjectStore((state) => state.updateImage);
  const setImageChannelPreview = useAppStore(
    (state) => state.setImageChannelPreview,
  );

  const channel = image.channels?.[c];
  const name = channel?.name ?? data.getChannelName?.(c) ?? `Channel ${c}`;
  const color = ImageUtils.getChannelColor(image, data, c);
  const contrastLimits = ImageUtils.getChannelContrastLimits(image, data, c);
  const visible = ImageUtils.getChannelVisibility(image, data, c);
  const opacity = ImageUtils.getChannelOpacity(image, data, c);

  const updateChannel = (updates: Partial<ImageChannel>) => {
    const channels = Array.from(
      { length: Math.max(c + 1, image.channels?.length ?? 0) },
      (_, i) => image.channels?.[i] ?? {},
    );
    channels[c] = { ...channels[c], ...updates };
    updateImage(image.id, { channels });
  };

  return (
    <Collapsible
      open={expanded}
      onOpenChange={onExpandedChange}
      disabled={contrastLimits === undefined}
      className={className}
    >
      <div className="flex flex-row items-center gap-x-1.5">
        <CollapsibleTriggerRightDownIcon />
        {image.channelViewMode !== ImageChannelViewMode.composite ? (
          <RadioGroupItem value={String(c)} aria-label={name} />
        ) : (
          <VisibilityButton
            visible={visible}
            onVisibleChange={(visibility) => {
              updateChannel({ visibility });
              setImageChannelPreview(null);
            }}
            objectLabel="channel"
            size="icon-xs"
            onPointerEnter={() =>
              setImageChannelPreview({ imageId: image.id, channelIndex: c })
            }
            onPointerLeave={() => setImageChannelPreview(null)}
          />
        )}
        {image.channelViewMode !== ImageChannelViewMode.grayscale ? (
          <ColorPicker
            color={color}
            onColorChange={(newColor: Color) =>
              updateChannel({ color: newColor })
            }
            label="Channel color"
            className="size-4 p-0 border-input shadow-xs"
          >
            <span
              className="block size-full rounded-sm"
              style={{ backgroundColor: ColorUtils.toHex(color) }}
            />
          </ColorPicker>
        ) : null}
        <CollapsibleTrigger className="flex-1 min-w-0">
          <span className="truncate" title={name}>
            {name}
          </span>
        </CollapsibleTrigger>
        <OpacityControl
          label="Channel opacity"
          opacity={opacity}
          onOpacityChange={(value) => updateChannel({ opacity: value })}
        />
      </div>
      {contrastLimits !== undefined ? (
        <CollapsiblePanel className="pt-1 pl-5">
          <ContrastRangeWidget
            contrastLimits={contrastLimits}
            histogram={data.getChannelHistogram?.(c)}
            dataTypeRange={data.getChannelDataTypeRange?.(c)}
            onContrastLimitsChange={(newContrastLimits) =>
              updateChannel({ contrastLimits: newContrastLimits })
            }
            onReset={
              channel?.contrastLimits !== undefined
                ? () => updateChannel({ contrastLimits: undefined })
                : undefined
            }
          />
        </CollapsiblePanel>
      ) : null}
    </Collapsible>
  );
}
