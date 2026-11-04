import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface BackendDocument {
  idDocument: number;
  titre: string;
  site?: string;
  source?: string;
  categorie?: string;
  fichier?: string;
  taille?: number;
  dateAjout?: string;
}

@Injectable({ providedIn: 'root' })
export class DocumentService {
  private readonly API = environment.apiUrl;

  constructor(private http: HttpClient) {}

  getAll(params?: {
    site?: string;
    source?: string;
    categorie?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }): Observable<BackendDocument[]> {
    return this.http.get<BackendDocument[]>(`${this.API}/documents`, { params: params as any });
  }

  getOne(id: number): Observable<BackendDocument> {
    return this.http.get<BackendDocument>(`${this.API}/documents/${id}`);
  }

  delete(id: number): Observable<any> {
    return this.http.delete(`${this.API}/documents/${id}`);
  }

  create(data: { titre: string; categorie: string; site?: string }): Observable<any> {
    return this.http.post(`${this.API}/documents`, data);
  }

  upload(formData: FormData): Observable<any> {
    return this.http.post(`${this.API}/documents/upload`, formData);
  }

  update(id: number, formData: FormData): Observable<any> {
    return this.http.put(`${this.API}/documents/${id}/file`, formData);
  }

  process(id: number): Observable<any> {
    return this.http.post(`${this.API}/documents/process/${id}`, {});
  }

  sync(): Observable<any> {
    return this.http.post(`${this.API}/documents/sync`, {});
  }

  getStats(): Observable<any> {
    return this.http.get(`${this.API}/documents/stats`);
  }
}
