import { Component, signal, inject, OnInit, effect } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule, FormBuilder, FormGroup } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';
import { HttpClient } from '@angular/common/http';
import { AuthService } from '../../../services/auth.service';
import { EmailService } from '../../../services/email.service';
import { animate, style, transition, trigger } from '@angular/animations';
import { environment } from '../../../../environments/environment';

@Component({
  selector: 'app-profile',
  standalone: true,
  imports: [CommonModule, FormsModule, ReactiveFormsModule, LucideAngularModule, TranslateModule],
  templateUrl: './profile.component.html',
  styleUrls: ['./profile.component.css'],
  animations: [
    trigger('fadeInOut', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(-20px)' }),
        animate('300ms ease-out', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
      transition(':leave', [
        animate('300ms ease-in', style({ opacity: 0, transform: 'translateY(-20px)' })),
      ]),
    ]),
  ],
})
export class ProfileComponent implements OnInit {
  private authService = inject(AuthService);
  private fb = inject(FormBuilder);
  private http = inject(HttpClient);
  private emailService = inject(EmailService);
  private readonly API = environment.apiUrl;

  user = this.authService.user;
  activeTab = signal<string>('informations');
  isEditing = signal<boolean>(false);
  showToast = signal<boolean>(false);
  avatarImage = signal<string | null>(null);
  showPassword = signal<boolean>(false);
  loading = signal(true);

  profileData = signal({
    id: 0,
    nom: '',
    prenom: '',
    email: '',
    motDePasse: '',
    telephone: '',
    dateNaissance: '',
    CIN: '',
    passport: '',
    pays: '',
    sexe: '',
    adresse: '',
    dernierDiplomeObtenu: '',
    anneeObtention: '',
    universite: '',
    diplomeSouhaite: '',
    niveau: '',
    cours: '',
    sourceConnaissanceUniversite: '',
  });

  editForm: FormGroup;

  tabs = [
    { id: 'informations', label: 'Informations' },
    { id: 'emails', label: 'Emails générés' },
  ];

  generatedEmails = signal<any[]>([]);

  constructor() {
    effect(() => {
      const u = this.user();
      if (u) {
        this.loadProfile(u.id);
      } else {
        this.loading.set(false);
      }
    }, { allowSignalWrites: true });

    this.editForm = this.fb.group({
      nom: [''],
      prenom: [''],
      email: [''],
      motDePasse: [''],
      telephone: [''],
      dateNaissance: [''],
      CIN: [''],
      passport: [''],
      pays: [''],
      sexe: [''],
      adresse: [''],
      dernierDiplomeObtenu: [''],
      anneeObtention: [''],
      universite: [''],
      diplomeSouhaite: [''],
      niveau: [''],
      cours: [''],
      sourceConnaissanceUniversite: [''],
    });
  }

  ngOnInit(): void {
    // Initial load is now handled by the effect in the constructor
  }

  private loadProfile(userId: number): void {
    this.http.get<any>(`${this.API}/etudiants/${userId}`).subscribe({
      next: (data) => {
        this.profileData.set({
          id: data.id,
          nom: data.nom || '',
          prenom: data.prenom || '',
          email: data.email || '',
          motDePasse: '',
          telephone: data.telephone || '',
          dateNaissance: data.dateNaissance || '',
          CIN: data.CIN || '',
          passport: data.passport || '',
          pays: data.pays || '',
          sexe: data.sexe || '',
          adresse: data.adresse || '',
          dernierDiplomeObtenu: data.dernierDiplomeObtenu || '',
          anneeObtention: data.anneeObtention || '',
          universite: data.universite || '',
          diplomeSouhaite: data.diplomeSouhaite || '',
          niveau: data.niveau || '',
          cours: data.cours || '',
          sourceConnaissanceUniversite: data.sourceConnaissanceUniversite || '',
        });
        if (data.photo) {
          this.avatarImage.set(data.photo);
        }
        this.loadEmails(data.id);
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }

  private loadEmails(etudiantId: number): void {
    this.emailService.getByStudent(etudiantId).subscribe({
      next: (emails) => this.generatedEmails.set(emails),
      error: () => {},
    });
  }

  initials(): string {
    const d = this.profileData();
    return ((d.prenom?.[0] || '') + (d.nom?.[0] || '')).toUpperCase().slice(0, 2);
  }

  formatDate(dateStr: string): string {
    if (!dateStr) return '';
    return new Date(dateStr).toLocaleDateString('fr-FR');
  }

  setActiveTab(tabId: string): void {
    this.activeTab.set(tabId);
  }

  startEdit(): void {
    this.editForm.patchValue(this.profileData());
    this.isEditing.set(true);
  }

  cancelEdit(): void {
    this.isEditing.set(false);
  }

  handleSave(): void {
    if (this.editForm.valid) {
      const rawData = this.editForm.value;
      const data: any = {};
      for (const key of Object.keys(rawData)) {
        data[key] = rawData[key] === '' ? null : rawData[key];
      }
      
      const avatarStr = this.avatarImage();
      if (avatarStr && avatarStr.startsWith('data:image')) {
        data['photo'] = avatarStr;
      }
      
      const userId = this.profileData().id;

      this.http.patch(`${this.API}/etudiants/${userId}`, data).subscribe({
        next: () => {
          this.profileData.update((d) => ({ ...d, ...data }));
          this.isEditing.set(false);
          this.showToast.set(true);
          if (avatarStr) {
            this.authService.updateAvatar(avatarStr);
          }
          this.showToast.set(true);
          setTimeout(() => this.showToast.set(false), 3000);
        },
        error: () => {},
      });
    }
  }

  handleImageUpload(event: any): void {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        this.avatarImage.set(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  }
}
