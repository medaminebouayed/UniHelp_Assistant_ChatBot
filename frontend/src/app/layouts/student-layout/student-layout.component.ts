import { Component, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule, RouterLinkActive, RouterLink } from '@angular/router';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';
import { AuthService } from '../../services/auth.service';
import { ChatbotFabComponent } from '../../components/chatbot-fab/chatbot-fab.component';

@Component({
  selector: 'app-student-layout',
  standalone: true,
  imports: [CommonModule, RouterModule, RouterLinkActive, RouterLink, LucideAngularModule, TranslateModule, ChatbotFabComponent],
  templateUrl: './student-layout.component.html',
  styles: [`
    .active-link {
      color: #1d4b99 !important;
    }
  `]
})
export class StudentLayoutComponent {
  private authService = inject(AuthService);
  private router = inject(Router);

  user = this.authService.user;
  isMobileMenuOpen = signal<boolean>(false);
  isProfileOpen = signal<boolean>(false);

  navItems = [
    { path: '/student/dashboard', labelKey: 'nav.dashboard', icon: 'layout-dashboard' },
    { path: '/student/emails', labelKey: 'nav.emails', icon: 'mail' },
    { path: '/student/faq', labelKey: 'nav.faq', icon: 'help-circle' }
  ];

  constructor() { }

  initials() {
    const u = this.user();
    if (!u || !u.name) return 'U';
    return u.name.split(' ').map((n: string) => n[0]).join('').toUpperCase().slice(0, 2);
  }

  toggleMobileMenu() {
    this.isMobileMenuOpen.update(v => !v);
  }

  toggleProfile() {
    this.isProfileOpen.update(v => !v);
  }

  navigateTo(path: string) {
    this.router.navigate([path]);
    this.isProfileOpen.set(false);
  }

  handleLogout() {
    this.authService.logout();
    this.router.navigate(['/login']);
  }
}
