/**
 * Gec gelen yanitlari ayiklar: yalnizca en son baslatilan istegin yaniti
 * islenir. Mod ya da set degisince invalidate() yoldaki istegi bayatlatir.
 */
export class RequestGate {
  private latest = 0;
  private active: number | null = null;

  begin(): number {
    this.latest += 1;
    this.active = this.latest;
    return this.latest;
  }

  isCurrent(token: number): boolean {
    return token === this.active;
  }

  finish(token: number): void {
    if (this.isCurrent(token)) this.active = null;
  }

  invalidate(): void {
    this.latest += 1;
    this.active = null;
  }

  /** Guncel istek henuz yanit almadi. */
  get pending(): boolean {
    return this.active !== null;
  }
}
