import { Component, Inject, PLATFORM_ID } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { TranslateService } from '@ngx-translate/core';
import { isPlatformBrowser } from '@angular/common';
import { ThemeService } from './services/theme.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  templateUrl: './app.component.html'
})
export class AppComponent {
  title = 'uniHelp';

  constructor(
    translate: TranslateService,
    theme: ThemeService,
    @Inject(PLATFORM_ID) platformId: object
  ) {
    translate.setDefaultLang('fr');
    if (isPlatformBrowser(platformId)) {
      const lang = localStorage.getItem('unihelp-lang');
      if (lang) {
        translate.use(lang);
        document.documentElement.lang = lang;
        document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
      }
    }

    theme.isDark();
  }
}