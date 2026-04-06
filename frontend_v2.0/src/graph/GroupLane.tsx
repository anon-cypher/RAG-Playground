import type { Node, NodeProps } from '@xyflow/react';

export type GroupLaneData = { label: string };

type GroupFlowNode = Node<GroupLaneData>;

/** Background lane for XYFlow grouping (indexing / query / generation). */
export function GroupLane({ data }: NodeProps<GroupFlowNode>) {
  return (
    <div className="group-lane">
      <span className="group-lane__title">{data.label}</span>
    </div>
  );
}
