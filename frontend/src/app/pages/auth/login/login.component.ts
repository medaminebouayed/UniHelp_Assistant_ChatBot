import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, Mail, Lock, Loader2 } from 'lucide-angular';
import { UniHelpLogoComponent } from '../../../components/uni-help-logo/uni-help-logo.component';
import { AuthService } from '../../../services/auth.service';
import { TranslateModule } from '@ngx-translate/core';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule, LucideAngularModule, UniHelpLogoComponent, TranslateModule],
  templateUrl: './login.component.html'
})
export class LoginComponent {
  email = '';
  password = '';
  error = signal<string>('');
  isSubmitting = signal<boolean>(false);
  showPassword = signal<boolean>(false);

  constructor(private authService: AuthService, private router: Router) {}

  async handleSubmit(e: Event) {
    e.preventDefault();
    this.error.set('');

    if (!this.email || !this.password) {
      this.error.set('Veuillez remplir tous les champs.');
      return;
    }

    this.isSubmitting.set(true);
    try {
      await this.authService.login(this.email, this.password);
      const user = this.authService.user();
      if (user?.role === 'admin') {
        this.router.navigate(['/admin/dashboard']);
      } else {
        this.router.navigate(['/student/dashboard']);
      }
    } catch (err: any) {
      if (err?.status === 401) {
        this.error.set('Email ou mot de passe incorrect.');
      } else {
        this.error.set('Erreur de connexion. Veuillez réessayer.');
      }
    } finally {
      this.isSubmitting.set(false);
    }
  }
}
