import { NODE_TYPES } from '../graph/nodeCatalog';
import type { NodeKind } from '../types/pipeline';

type PaletteProps = {
  onDragStart: (kind: NodeKind, e: React.DragEvent) => void;
  onAddClick: (kind: NodeKind) => void;
};

export function Palette({ onDragStart, onAddClick }: PaletteProps) {
  return (
    <aside className="palette">
      <h3 className="palette__title">Components</h3>
      <p className="palette__hint">Drag onto canvas or click +</p>
      <ul className="palette__list">
        {NODE_TYPES.map((t) => (
          <li key={t.kind}>
            <button
              type="button"
              className="palette__item"
              draggable
              onDragStart={(e) => onDragStart(t.kind, e)}
              onClick={() => onAddClick(t.kind)}
              title={t.description}
            >
              <span className="palette__label">{t.label}</span>
              <span className="palette__kind">{t.kind}</span>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
}
