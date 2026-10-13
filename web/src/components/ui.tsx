import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { Link } from 'react-router-dom';
import Icon, { type IconName } from './Icon';

type Gradient = 'purple' | 'green' | 'orange';

/** Ana sayfadaki renk gecisli buyuk dugme (Flutter _BigButton). */
export function BigButton(props: { to: string; label: string; sublabel: string; icon: IconName; gradient: Gradient }) {
  return (
    <Link to={props.to} className={`big-button big-button--${props.gradient}`}>
      <span className="big-button-icon">
        <Icon name={props.icon} size={28} />
      </span>
      <span className="big-button-text">
        <span className="big-button-label">{props.label}</span>
        <span className="big-button-sublabel">{props.sublabel}</span>
      </span>
      <Icon name="arrow_forward_ios" size={16} />
    </Link>
  );
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & { icon?: IconName; children: ReactNode };

function ButtonContent({ icon, children }: { icon?: IconName; children: ReactNode }) {
  return (
    <>
      {icon && <Icon name={icon} />}
      <span>{children}</span>
    </>
  );
}

/** Dolu, ikonlu dugme (Flutter FilledButton.icon). Renk icin className: 'button--danger', 'button--settings'. */
export function FilledButton({ icon, children, className = '', type = 'button', ...rest }: ButtonProps) {
  return (
    <button type={type} className={`button button--filled ${className}`} {...rest}>
      <ButtonContent icon={icon}>{children}</ButtonContent>
    </button>
  );
}

/** Cerceveli, ikonlu dugme (Flutter OutlinedButton.icon). */
export function OutlinedButton({ icon, children, className = '', type = 'button', ...rest }: ButtonProps) {
  return (
    <button type={type} className={`button button--outlined ${className}`} {...rest}>
      <ButtonContent icon={icon}>{children}</ButtonContent>
    </button>
  );
}

/** Kucuk yuvarlak ikon dugmesi (oynat, sil). */
export function IconButton({ icon, label, ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & { icon: IconName; label: string }) {
  return (
    <button type="button" className="icon-button" aria-label={label} title={label} {...rest}>
      <Icon name={icon} size={20} />
    </button>
  );
}

/** Renkli cerceveli bilgi kutusu (Flutter _ResultTile). Hata kutusu ekran okuyucuya hemen bildirilir. */
export function ResultTile({ label, value, tone = 'accent' }: { label: string; value: ReactNode; tone?: 'accent' | 'settings' | 'error' }) {
  return (
    <div className={`result-tile result-tile--${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <div className="result-tile-label">{label}</div>
      <div className="result-tile-value">{value}</div>
    </div>
  );
}

export function SectionHeader({ children }: { children: ReactNode }) {
  return <h2 className="section-header">{children}</h2>;
}

/** Yuzde (0-100) ile dolan cubuk (Flutter LinearProgressIndicator). */
export function ProgressBar({ value, label }: { value: number; label?: string }) {
  const width = Math.min(100, Math.max(0, value));
  return (
    <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(width)} aria-label={label}>
      <div className="progress-fill" style={{ width: `${width}%` }} />
    </div>
  );
}

/** Secili parcasi dolu segment dugmesi (Flutter SegmentedButton). */
export function SegmentedButton<T extends string>(props: {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <div className={`segmented ${props.className ?? ''}`} role="group" aria-label={props.label}>
      {props.options.map((o) => (
        <button
          key={o.value}
          type="button"
          className="segmented-option"
          aria-pressed={o.value === props.value}
          disabled={props.disabled}
          onClick={() => props.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
