import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import Icon from './Icon';

/**
 * Flutter AppBar'i: geri oku sabit bir sayfaya gider (gecmis bos olsa da calisir).
 * replace: gecmis buyumez, ana sayfada telefonun geri tusu uygulamayi kapatir.
 */
export default function AppBar({ title, backTo, action }: { title: string; backTo: string; action?: ReactNode }) {
  return (
    <header className="app-bar">
      <Link to={backTo} replace className="icon-button" aria-label="Geri">
        <Icon name="arrow_back" />
      </Link>
      <h1 className="app-bar-title">{title}</h1>
      {action && <div className="app-bar-action">{action}</div>}
    </header>
  );
}
