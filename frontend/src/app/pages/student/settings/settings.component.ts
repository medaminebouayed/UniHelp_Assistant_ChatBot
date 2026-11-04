import { Component, signal, WritableSignal, Inject, PLATFORM_ID } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { LucideAngularModule } from 'lucide-angular';
import { animate, style, transition, trigger } from '@angular/animations';
import { TranslateModule, TranslateService } from '@ngx-translate/core';
import { ThemeService } from '../../../services/theme.service';

interface Language {
  id: string;
  label: string;
  flag: string;
}

@Component({
  selector: 'app-settings',
  standalone: true,
  imports: [CommonModule, LucideAngularModule, TranslateModule],
  templateUrl: './settings.component.html',
  animations: [
    trigger('scaleIn', [
      transition(':enter', [
        style({ transform: 'scale(0)' }),
        animate('200ms ease-out', style({ transform: 'scale(1)' }))
      ])
    ]),
    trigger('fadeIn', [
      transition(':enter', [
        style({ opacity: 0 }),
        animate('300ms ease-out', style({ opacity: 1 }))
      ]),
      transition(':leave', [
        animate('200ms ease-in', style({ opacity: 0 }))
      ])
    ]),
    trigger('modalAnimation', [
      transition(':enter', [
        style({ opacity: 0, transform: 'scale(0.9) translateY(30px)' }),
        animate('400ms cubic-bezier(0.34, 1.56, 0.64, 1)', style({ opacity: 1, transform: 'scale(1) translateY(0)' }))
      ]),
      transition(':leave', [
        animate('200ms ease-in', style({ opacity: 0, transform: 'scale(0.9) translateY(30px)' }))
      ])
    ])
  ]
})
export class SettingsComponent {
  selectedLang = signal<string>('fr');
  isDarkMode: WritableSignal<boolean>;
  isAboutModalOpen = signal<boolean>(false);

  languages: Language[] = [
    { id: 'fr', label: 'Français', flag: '🇫🇷' },
    { id: 'en', label: 'English', flag: '🇬🇧' },
    { id: 'ar', label: 'العربية', flag: '🇹🇳' }
  ];

  constructor(
    private translate: TranslateService,
    public theme: ThemeService,
    @Inject(PLATFORM_ID) private platformId: object
  ) {
    this.isDarkMode = theme.isDark;
    const lang = translate.currentLang || translate.getDefaultLang();
    this.selectedLang.set(lang || 'fr');
  }

  setSelectedLang(id: string) {
    this.selectedLang.set(id);
    this.translate.use(id);
    if (isPlatformBrowser(this.platformId)) {
      localStorage.setItem('unihelp-lang', id);
      document.documentElement.lang = id;
      document.documentElement.dir = id === 'ar' ? 'rtl' : 'ltr';
    }
  }

  toggleDarkMode() {
    this.theme.toggle();
  }

  openAbout() {
    this.isAboutModalOpen.set(true);
  }

  closeAbout() {
    this.isAboutModalOpen.set(false);
  }
}