import { Component, signal, inject, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';
import { AuthService } from '../../../services/auth.service';
import { EmailService } from '../../../services/email.service';
import { animate, style, transition, trigger, query, stagger } from '@angular/animations';

@Component({
  selector: 'app-email-generator',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule, TranslateModule],
  templateUrl: './email-generator.component.html',
  styleUrls: ['./email-generator.component.css']
})
export class EmailGeneratorComponent {
  private authService = inject(AuthService);
  private emailService = inject(EmailService);

  user = this.authService.user;
  step = signal<number>(1);
  isGenerating = signal<boolean>(false);
  isCopied = signal<boolean>(false);
  generatedEmail = signal<string>('');

  formData = signal({
    type: '',
    recipient: '',
    motif: ''
  });

  emailTypes = [
    { value: '', label: "Sélectionnez un type d'email..." },
    { value: 'attestation', label: "Demande d'attestation de scolarité" },
    { value: 'rattrapage', label: 'Demande de rattrapage' },
    { value: 'stage', label: 'Demande de stage' },
    { value: 'info', label: "Demande d'information" },
    { value: 'reclamation', label: 'Réclamation' },
    { value: 'remerciement', label: 'Remerciement' }
  ];

  recipients = [
    { value: '', label: 'Sélectionnez un destinataire...' },
    { value: 'directeur', label: "Directeur de l'établissement" },
    { value: 'chef_dept', label: 'Chef de département' },
    { value: 'scolarite', label: 'Service scolarité' },
    { value: 'enseignant', label: 'Enseignant' }
  ];

  steps = [
    { id: 1, label: 'Type' },
    { id: 2, label: 'Destinataire' },
    { id: 3, label: 'Infos' },
    { id: 4, label: 'Aperçu' }
  ];

  emailBlocks = computed(() => {
    const email = this.generatedEmail();
    if (!email) return [];

    const studentName = this.user()?.name || 'Jean Dupont';
    const studentField = 'Informatique'; // Simplified for now

    return email.split('\n').map(line => {
      if (line.startsWith('À :') || line.startsWith('À:')) {
        return { type: 'header', content: line };
      }
      if (line.startsWith('Objet :') || line.startsWith('Objet:')) {
        return { type: 'subject', content: line };
      }
      if (line.trim() === '') {
        return { type: 'br' };
      }

      // Handle bold parts
      const parts = [];
      const boldTerms = [studentName, `2ème année ${studentField}`];
      let currentLine = line;

      // Simplistic splitting for demonstration
      // In a real app, use a more robust regex splitter
      parts.push({ text: line, bold: false });

      return { type: 'body', content: line, parts: parts };
    });
  });

  isStepValid(s: number): boolean {
    const data = this.formData();
    if (s === 1) return data.type !== '';
    if (s === 2) return data.recipient !== '';
    if (s === 3) return data.motif.trim() !== '';
    return true;
  }

  handleNext() {
    if (this.isStepValid(this.step())) {
      this.step.update(s => s + 1);
    }
  }

  handleBack() {
    if (this.step() > 1) {
      this.step.update(s => s - 1);
    }
  }

  reset() {
    this.step.set(1);
    this.formData.set({ type: '', recipient: '', motif: '' });
    this.generatedEmail.set('');
  }

  async generate() {
    this.isGenerating.set(true);
    const user = this.user();
    if (!user) return;

    const data = this.formData();
    try {
      const res = await this.emailService
        .generate({
          etudiant_id: user.id,
          type_email: data.type,
          destinataire: this.getRecipientEmail(data.recipient),
          details: data.motif,
        })
        .toPromise();

      this.generatedEmail.set(res?.contenu || '');
      this.step.set(4);
    } catch {
      this.generatedEmail.set('Erreur lors de la génération de l\'email. Veuillez réessayer.');
      this.step.set(4);
    } finally {
      this.isGenerating.set(false);
    }
  }

  private getRecipientEmail(r: string): string {
    const emails: Record<string, string> = {
      directeur: 'direction@universite.tn',
      chef_dept: 'chef-departement@universite.tn',
      scolarite: 'scolarite@universite.tn',
      enseignant: 'professeur@universite.tn',
    };
    return emails[r] || '';
  }

  getRecipientTitle(r: string): string {
    if (r === 'directeur') return 'Monsieur le Directeur';
    if (r === 'chef_dept') return 'Monsieur le Chef de Département';
    if (r === 'scolarite') return 'Madame, Monsieur';
    if (r === 'enseignant') return 'Madame, Monsieur le Professeur';
    return 'Madame, Monsieur';
  }

  copyToClipboard() {
    navigator.clipboard.writeText(this.generatedEmail());
    this.isCopied.set(true);
    setTimeout(() => this.isCopied.set(false), 2000);
  }

  isSentModalOpen = signal<boolean>(false);

  sendEmail() {
    // In a real application, call email service to send the email
    this.isSentModalOpen.set(true);
  }

  closeSentModal() {
    this.isSentModalOpen.set(false);
    this.reset();
  }
}
