import { Injectable, signal, effect } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { environment } from '../../environments/environment';

export type Role = 'student' | 'admin';

export interface User {
  id: number;
  nom: string;
  prenom: string;
  name?: string;
  email: string;
  role: Role;
  avatar?: string;
  universite?: string;
  diplomeSouhaite?: string;
  telephone?: string;
  departement?: string;
}

export interface LoginResponse {
  access_token: string;
}

export interface MeResponse {
  id: number;
  nom: string;
  prenom: string;
  email: string;
  role: string;
  photo?: string | null;
  universite?: string | null;
  diplomeSouhaite?: string | null;
  telephone?: string | null;
  departement?: string | null;
}

export interface RegisterData {
  nom: string;
  prenom: string;
  email: string;
  motDePasse: string;
  CIN?: string;
  passport?: string;
  pays?: string;
  dateNaissance?: string;
  sexe?: string;
  telephone?: string;
  adresse?: string;
  dernierDiplomeObtenu?: string;
  anneeObtention?: string;
  universite?: string;
  diplomeSouhaite?: string;
  niveau?: string;
  cours?: string;
  sourceConnaissanceUniversite?: string;
}

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  private readonly API = environment.apiUrl;

  user = signal<User | null>(null);
  isLoading = signal<boolean>(true);

  constructor(private http: HttpClient, private router: Router) {
    const token = localStorage.getItem('access_token');
    if (token) {
      this.fetchMe();
    } else {
      this.isLoading.set(false);
    }
  }

  private fetchMe(): void {
    this.http.get<MeResponse>(`${this.API}/auth/me`).subscribe({
      next: (res) => {
        this.user.set(this.mapMeToUser(res));
        this.isLoading.set(false);
      },
      error: () => {
        localStorage.removeItem('access_token');
        this.isLoading.set(false);
      },
    });
  }

  private mapMeToUser(res: MeResponse): User {
    return {
      id: res.id,
      nom: res.nom,
      prenom: res.prenom,
      name: `${res.prenom} ${res.nom}`,
      email: res.email,
      role: res.role === 'etudiant' ? 'student' : 'admin',
      avatar: res.photo || `https://ui-avatars.com/api/?name=${res.prenom}+${res.nom}&background=random`,
      universite: res.universite || undefined,
      diplomeSouhaite: res.diplomeSouhaite || undefined,
      telephone: res.telephone || undefined,
      departement: res.departement || undefined,
    };
  }

  login(email: string, password: string): Promise<void> {
    this.isLoading.set(true);
    return new Promise((resolve, reject) => {
      this.http
        .post<LoginResponse>(`${this.API}/auth/login`, {
          email,
          motDePasse: password,
        })
        .subscribe({
          next: (res) => {
            localStorage.setItem('access_token', res.access_token);
            this.http.get<MeResponse>(`${this.API}/auth/me`).subscribe({
              next: (me) => {
                this.user.set(this.mapMeToUser(me));
                this.isLoading.set(false);
                resolve();
              },
              error: (err) => {
                this.isLoading.set(false);
                reject(err);
              },
            });
          },
          error: (err) => {
            this.isLoading.set(false);
            reject(err);
          },
        });
    });
  }

  register(data: RegisterData): Promise<void> {
    this.isLoading.set(true);
    return new Promise((resolve, reject) => {
      this.http
        .post<{ id: number; email: string; role: string }>(
          `${this.API}/auth/register`,
          data
        )
        .subscribe({
          next: (res) => {
            // Auto-login after registration
            this.login(data.email, data.motDePasse).then(resolve).catch(reject);
          },
          error: (err) => {
            this.isLoading.set(false);
            reject(err);
          },
        });
    });
  }

  logout(): void {
    this.user.set(null);
    localStorage.removeItem('access_token');
    this.router.navigate(['/login']);
  }

  updateAvatar(avatarUrl: string): void {
    this.user.update(u => u ? { ...u, avatar: avatarUrl } : null);
  }
}
