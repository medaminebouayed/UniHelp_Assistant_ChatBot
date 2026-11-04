import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface BackendEmail {
  idEmail: number;
  object: string;
  destinataire: string;
  contenu: string;
  etudiant_id: number;
}

export interface GenerateEmailRequest {
  etudiant_id: number;
  type_email: string;
  destinataire?: string;
  details?: string;
}

export interface GenerateEmailResponse extends BackendEmail {
  send_success: boolean;
  send_error?: string | null;
}

@Injectable({ providedIn: 'root' })
export class EmailService {
  private readonly API = environment.apiUrl;

  constructor(private http: HttpClient) {}

  generate(body: GenerateEmailRequest): Observable<GenerateEmailResponse> {
    return this.http.post<GenerateEmailResponse>(
      `${this.API}/emails-generes/generer`,
      body
    );
  }

  getByStudent(etudiant_id: number): Observable<BackendEmail[]> {
    return this.http.get<BackendEmail[]>(
      `${this.API}/emails-generes/etudiant/${etudiant_id}`
    );
  }

  delete(id: number): Observable<any> {
    return this.http.delete(`${this.API}/emails-generes/${id}`);
  }
}
