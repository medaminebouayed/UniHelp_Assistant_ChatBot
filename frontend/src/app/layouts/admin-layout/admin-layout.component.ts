import { Component, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule, RouterLinkActive, RouterLink } from '@angular/router';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-admin-layout',
  standalone: true,
  imports: [CommonModule, RouterModule, RouterLinkActive, RouterLink, LucideAngularModule, TranslateModule],
  templateUrl: './admin-layout.component.html',
  styles: [`
    .active-link {
      color: #1d4b99 !important;
    }
  `]
})
export class AdminLayoutComponent {
  private authService = inject(AuthService);
  private router = inject(Router);

  user = this.authService.user;
  isMobileMenuOpen = signal<boolean>(false);
  isProfileOpen = signal<boolean>(false);

  navItems = [
    { path: '/admin/dashboard', labelKey: 'admin.dashboard', icon: 'layout-dashboard' },
    { path: '/admin/documents', labelKey: 'admin.documents', icon: 'file-text' },
    { path: '/admin/students', labelKey: 'admin.students', icon: 'users' },
  ];

  initials() {
    const u = this.user();
    if (!u || !u.name) return 'AD';
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
