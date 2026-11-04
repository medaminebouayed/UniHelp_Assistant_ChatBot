import { Component, signal, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';
import { Router } from '@angular/router';
import { animate, style, transition, trigger, state } from '@angular/animations';

interface Question {
  q: string;
  a: string;
}

interface FaqSection {
  category: string;
  icon: string;
  questions: Question[];
}

@Component({
  selector: 'app-faq',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule, TranslateModule],
  templateUrl: './faq.component.html',
  animations: [
    trigger('expandCollapse', [
      state('collapsed', style({ height: '0px', opacity: 0, visibility: 'hidden' })),
      state('expanded', style({ height: '*', opacity: 1, visibility: 'visible' })),
      transition('expanded <=> collapsed', animate('250ms ease-in-out'))
    ])
  ]
})
export class FaqComponent {
  private router = inject(Router);

  searchQuery = signal<string>('');
  activeCategory = signal<string>('Tous');
  openIndex = signal<string | null>(null);

  faqData: FaqSection[] = [
    {
      category: 'Inscription',
      icon: 'clipboard-list',
      questions: [
        {
          q: "Comment puis-je m'inscrire aux cours du semestre prochain ?",
          a: "Les inscriptions pédagogiques se font via le portail étudiant dans la section 'Mes Cours'. Vous recevrez un email avec les dates exactes d'ouverture des inscriptions."
        },
        {
          q: 'Où puis-je trouver mon certificat de scolarité ?',
          a: "Votre certificat de scolarité est disponible dans la section 'Documents' de votre profil, une fois votre inscription administrative validée."
        }
      ]
    },
    {
      category: 'Cours',
      icon: 'book-open',
      questions: [
        {
          q: 'Comment accéder aux supports de cours ?',
          a: "Les supports de cours sont mis en ligne par vos professeurs dans la section 'Documents' ou directement sur la plateforme Moodle de l'université."
        },
        {
          q: "Que faire en cas d'absence à un TD/TP ?",
          a: 'Toute absence doit être justifiée dans les 48h auprès du secrétariat de votre filière avec un justificatif médical ou officiel.'
        }
      ]
    },
    {
      category: 'Technique',
      icon: 'settings',
      questions: [
        {
          q: "J'ai oublié mon mot de passe, comment le réinitialiser ?",
          a: "Sur la page de connexion, cliquez sur 'Mot de passe oublié'. Un lien de réinitialisation vous sera envoyé sur votre adresse email de récupération."
        },
        {
          q: 'Comment fonctionne le Chatbot IA ?',
          a: "Le Chatbot IA est conçu pour répondre à vos questions sur l'administration et vous aider à résumer vos cours. Il suffit de lui poser une question naturellement dans la section 'Chatbot IA'."
        }
      ]
    },
    {
      category: 'Administration',
      icon: 'building',
      questions: [
        {
          q: 'Comment contacter le secrétariat ?',
          a: "Vous pouvez utiliser le 'Générateur d'emails' pour formuler votre demande, puis l'envoyer à l'adresse email de votre secrétariat (disponible dans l'annuaire)."
        },
        {
          q: 'Où puis-je consulter mes notes ?',
          a: "Les notes sont publiées sur le portail étudiant (ENT) à la fin de chaque période d'examen, après délibération du jury."
        }
      ]
    }
  ];

  categories = ['Tous', ...this.faqData.map(d => d.category)];

  normalize(str: string): string {
    return str.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  }

  filteredFaqs = computed(() => {
    const query = this.normalize(this.searchQuery());
    const activeCat = query ? 'Tous' : this.activeCategory();

    return this.faqData
      .map(section => {
        const filteredQuestions = section.questions.filter(
          q => this.normalize(q.q).includes(query) || this.normalize(q.a).includes(query)
        );
        return { ...section, questions: filteredQuestions };
      })
      .filter(
        section => (activeCat === 'Tous' || section.category === activeCat) && section.questions.length > 0
      );
  });

  setActiveCategory(category: string) {
    this.activeCategory.set(category);
    this.openIndex.set(null);
  }

  toggleAccordion(sIdx: number, qIdx: number) {
    const id = `${sIdx}-${qIdx}`;
    this.openIndex.set(this.openIndex() === id ? null : id);
  }

  isOpen(sIdx: number, qIdx: number): boolean {
    return this.openIndex() === `${sIdx}-${qIdx}`;
  }

  navigateToChat() {
    this.router.navigate(['/student/chatbot']);
  }
}
