import { Component, signal, computed, inject, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';
import { AuthService } from '../../../services/auth.service';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../../environments/environment';
import { animate, style, transition, trigger, state } from '@angular/animations';

@Component({
  selector: 'app-admin-profile',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule, TranslateModule],
  templateUrl: './profile.component.html',
  animations: [
    trigger('toastAnimation', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(-20px)' }),
        animate('300ms ease-out', style({ opacity: 1, transform: 'translateY(0)' }))
      ]),
      transition(':leave', [
        animate('300ms ease-in', style({ opacity: 0, transform: 'translateY(-20px)' }))
      ])
    ]),
    trigger('tabTransition', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(10px)' }),
        animate('300ms ease-out', style({ opacity: 1, transform: 'translateY(0)' }))
      ])
    ])
  ]
})
export class AdminProfileComponent {
  private authService = inject(AuthService);
  private http = inject(HttpClient);
  private readonly API = environment.apiUrl;

  @ViewChild('avatarInput') avatarInput!: ElementRef<HTMLInputElement>;

  user = this.authService.user;
  isEditing = signal(false);
  activeTab = signal('informations');
  showToast = signal(false);
  avatarImage = signal<string | null>(null);

  formData = signal({
    nom: '',
    prenom: '',
    email: '',
    motDePasse: '********',
    phone: '',
    role: '',
    department: ''
  });

  editData = { ...this.formData() };

  initials = computed(() => {
    const f = this.formData();
    return ((f.prenom?.[0] || 'A') + (f.nom?.[0] || 'D')).toUpperCase().slice(0, 2);
  });

  tabs = [
    { id: 'informations', label: 'Informations' },
    { id: 'activite', label: 'Activité récente' }
  ];

  recentActivities: any[] = [];

  constructor() {
    const u = this.user();
    if (u) {
      this.formData.set({
        nom: u.name?.split(' ').slice(1).join(' ') || 'Admin',
        prenom: u.name?.split(' ')[0] || 'Admin',
        email: u.email || '',
        motDePasse: '********',
        phone: (u as any).telephone || '+216 XX XXX XXX',
        role: 'Administrateur',
        department: (u as any).departement || 'Direction Générale',
      });
      this.editData = { ...this.formData() };
    }
    this.loadActivity();
  }

  private loadActivity(): void {
    this.http.get(`${this.API}/dashboard/stats`).subscribe({
      next: (res: any) => {
        if (res.recent_activity) {
          this.recentActivities = res.recent_activity;
        }
      },
      error: () => {},
    });
  }

  onAvatarChange(event: any) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        this.avatarImage.set(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  }

  handleSave() {
    const u = this.user();
    if (!u) return;
    const payload = { ...this.editData };
    if (this.avatarImage() && this.avatarImage()!.startsWith('data:image')) {
      (payload as any).photo = this.avatarImage();
    }
    
    this.http.patch(`${this.API}/administrateurs/${u.id}`, payload).subscribe({
      next: (res: any) => {
        this.formData.set({ ...this.editData });
        this.isEditing.set(false);
        this.showToast.set(true);
        if (this.avatarImage()) {
          this.authService.updateAvatar(this.avatarImage()!);
        }
        setTimeout(() => this.showToast.set(false), 3000);
      },
      error: () => {}
    });
  }
}
