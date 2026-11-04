import { Injectable, signal, effect, Inject, PLATFORM_ID } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  isDark = signal<boolean>(false);

  constructor(@Inject(PLATFORM_ID) private platformId: object) {
    if (isPlatformBrowser(this.platformId)) {
      const saved = localStorage.getItem('unihelp-dark-mode');
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const initial = saved !== null ? saved === 'true' : prefersDark;
      this.isDark.set(initial);
      this.apply(initial);
    }
  }

  toggle() {
    const next = !this.isDark();
    this.isDark.set(next);
    if (isPlatformBrowser(this.platformId)) {
      localStorage.setItem('unihelp-dark-mode', String(next));
    }
    this.apply(next);
  }

  private apply(dark: boolean) {
    if (isPlatformBrowser(this.platformId)) {
      document.documentElement.classList.toggle('dark', dark);
    }
  }
}