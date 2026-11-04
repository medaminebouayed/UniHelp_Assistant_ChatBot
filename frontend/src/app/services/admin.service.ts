import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface BackendAdmin {
  id: number;
  nom: string;
  prenom: string;
  email: string;
  telephone?: string;
  departement?: string;
  documents: any[];
}

export interface CreateAdminRequest {
  nom: string;
  prenom: string;
  email: string;
  motDePasse: string;
  telephone?: string;
  departement?: string;
}

@Injectable({ providedIn: 'root' })
export class AdminService {
  private readonly API = environment.apiUrl;

  constructor(private http: HttpClient) {}

  getAll(): Observable<BackendAdmin[]> {
    return this.http.get<BackendAdmin[]>(`${this.API}/administrateurs`);
  }

  getOne(id: number): Observable<BackendAdmin> {
    return this.http.get<BackendAdmin>(`${this.API}/administrateurs/${id}`);
  }

  create(body: CreateAdminRequest): Observable<BackendAdmin> {
    return this.http.post<BackendAdmin>(`${this.API}/administrateurs`, body);
  }

  delete(id: number): Observable<any> {
    return this.http.delete(`${this.API}/administrateurs/${id}`);
  }
}
