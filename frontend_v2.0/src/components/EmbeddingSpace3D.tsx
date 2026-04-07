import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useMemo } from 'react';
import * as THREE from 'three';
import type { Point3D, VisualizationData } from '../types/pipeline';

const MAX_EDGE_SEGMENTS = 400;

function computeCenter(data: VisualizationData): [number, number, number] {
  const pts: Point3D[] = [...data.points];
  if (data.query_point) pts.push(data.query_point);
  if (pts.length === 0) return [0, 0, 0];
  let sx = 0;
  let sy = 0;
  let sz = 0;
  for (const p of pts) {
    sx += p.x;
    sy += p.y;
    sz += p.z;
  }
  const n = pts.length;
  return [sx / n, sy / n, sz / n];
}

function positionForIndex(
  data: VisualizationData,
  idx: number,
  offset: [number, number, number],
): [number, number, number] | null {
  const { points, query_point } = data;
  const o = offset;
  if (idx === points.length && query_point) {
    return [query_point.x - o[0], query_point.y - o[1], query_point.z - o[2]];
  }
  if (idx >= 0 && idx < points.length) {
    const p = points[idx];
    return [p.x - o[0], p.y - o[1], p.z - o[2]];
  }
  return null;
}

/** Single THREE.LineSegments for all query→neighbor edges (reliable vs many drei Line meshes). */
function EdgeLines({
  data,
  offset,
}: {
  data: VisualizationData;
  offset: [number, number, number];
}) {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const { edges } = data;
    const cap = Math.min(edges.length, MAX_EDGE_SEGMENTS);
    const verts: number[] = [];
    for (let e = 0; e < cap; e++) {
      const edge = edges[e];
      if (!edge || edge.length < 2) continue;
      const [a, b] = edge;
      const pa = positionForIndex(data, a, offset);
      const pb = positionForIndex(data, b, offset);
      if (!pa || !pb) continue;
      verts.push(pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]);
    }
    if (verts.length === 0) return null;
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    return geo;
  }, [data, offset]);

  if (!geometry) return null;

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#a78bfa" transparent opacity={0.9} depthTest />
    </lineSegments>
  );
}

export type VizSelection =
  | { type: 'chunk'; index: number }
  | { type: 'query' }
  | null;

function Scene({
  data,
  offset,
  onSelect,
}: {
  data: VisualizationData;
  offset: [number, number, number];
  onSelect: (sel: VizSelection) => void;
}) {
  const qPos = data.query_point
    ? ([
        data.query_point.x - offset[0],
        data.query_point.y - offset[1],
        data.query_point.z - offset[2],
      ] as [number, number, number])
    : null;

  return (
    <>
      <ambientLight intensity={0.4} />

      {/* Subtle grid for depth / scale (does not affect meshBasicMaterial points) */}
      <gridHelper args={[80, 20, '#334155', '#1e293b']} position={[0, -12, 0]} />

      {data.points.map((p, i) => (
        <mesh
          key={p.chunk_id || `c-${i}`}
          position={[p.x - offset[0], p.y - offset[1], p.z - offset[2]]}
          onClick={(e) => {
            e.stopPropagation();
            onSelect({ type: 'chunk', index: i });
          }}
        >
          <sphereGeometry args={[0.22, 14, 14]} />
          <meshBasicMaterial color={p.is_neighbor ? '#22d3ee' : '#64748b'} />
        </mesh>
      ))}

      {qPos && (
        <mesh position={qPos} onClick={(e) => { e.stopPropagation(); onSelect({ type: 'query' }); }}>
          <sphereGeometry args={[0.42, 18, 18]} />
          <meshBasicMaterial color="#fbbf24" />
        </mesh>
      )}

      <EdgeLines data={data} offset={offset} />

      <OrbitControls enableDamping makeDefault minDistance={6} maxDistance={160} />
    </>
  );
}

export type EmbeddingSpace3DProps = {
  data: VisualizationData | null;
  onSelect: (sel: VizSelection) => void;
};

export function EmbeddingSpace3D({ data, onSelect }: EmbeddingSpace3DProps) {
  const offset = useMemo(() => (data ? computeCenter(data) : [0, 0, 0]), [data]) as [
    number,
    number,
    number,
  ];

  if (!data || (data.points.length === 0 && !data.query_point)) {
    return (
      <div className="embedding-viz embedding-viz--empty">
        <p>No projection data. Run a query with an indexed corpus, or load embedding space.</p>
      </div>
    );
  }

  return (
    <div className="embedding-viz">
      <Canvas
        style={{ width: '100%', height: '100%', display: 'block' }}
        camera={{ position: [0, 8, 38], fov: 50, near: 0.1, far: 2000 }}
        gl={{ antialias: true, alpha: false }}
        dpr={[1, 2]}
      >
        <color attach="background" args={['#050810']} />
        <Scene data={data} offset={offset} onSelect={onSelect} />
      </Canvas>
      <p className="embedding-viz__disclaimer">
        <strong>Note:</strong> The 3D view is a PCA projection for intuition only. Distances in this plot are
        not the same as ranking distance in the original embedding space.
      </p>
      <p className="embedding-viz__hint">
        Drag to rotate · scroll to zoom · click points for chunk details.
      </p>
    </div>
  );
}
