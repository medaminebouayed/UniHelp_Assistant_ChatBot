import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { trigger, transition, animate, style } from '@angular/animations';
import { UniHelpLogoComponent } from '../../../components/uni-help-logo/uni-help-logo.component';
import { AuthLayoutComponent } from '../../../layouts/auth-layout/auth-layout.component';
import { AuthService } from '../../../services/auth.service';
import { TranslateModule } from '@ngx-translate/core';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule, LucideAngularModule, UniHelpLogoComponent, AuthLayoutComponent, TranslateModule],
  animations: [
    trigger('stepTransition', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateX(20px)' }),
        animate('0.3s ease-out', style({ opacity: 1, transform: 'translateX(0)' }))
      ]),
      transition(':leave', [
        animate('0.3s ease-in', style({ opacity: 0, transform: 'translateX(-20px)' }))
      ])
    ])
  ],
  templateUrl: './register.component.html'
})
export class RegisterComponent {
  step = signal<number>(1);
  error = signal<string>('');
  isSubmitting = signal<boolean>(false);
  showPassword = signal<boolean>(false);
  showCountryDropdown = signal<boolean>(false);
  showSexDropdown = signal<boolean>(false);
  showLastDiplomaDropdown = signal<boolean>(false);

  showUniversityDropdown = signal<boolean>(false);
  showDesiredDiplomaDropdown = signal<boolean>(false);
  showLevelDropdown = signal<boolean>(false);
  showCourseTimeDropdown = signal<boolean>(false);
  showKnownByDropdown = signal<boolean>(false);

  countrySearch = '';
  sexSearch = '';
  lastDiplomaSearch = '';
  universitySearch = '';
  desiredDiplomaSearch = '';
  levelSearch = '';
  courseTimeSearch = '';
  knownBySearch = '';

  sexOptions = ['Homme', 'Femme'];
  diplomaOptions = ['Baccalauréat', 'Prépa', 'Licence', 'Master'];
  universityOptions = ['ISB', 'IIT'];
  levelOptions = ['1', '2', '3'];
  courseTimeOptions = ['Cours de jour', 'Cours de soir'];
  knownByOptions = [
    { label: 'Réseaux sociaux', icon: 'globe' },
    { label: 'Amis / Parents', icon: 'users' },
    { label: 'Site web', icon: 'globe' },
    { label: 'SMS', icon: 'message-square' },
    { label: 'Press', icon: 'file-text' },
    { label: 'Foire', icon: 'building' },
    { label: 'Tourné Lycée', icon: 'map-pin' },
    { label: 'Événement', icon: 'calendar' },
    { label: 'Email', icon: 'mail' },
    { label: 'Radio/TV', icon: 'megaphone' },
    { label: 'Autre', icon: 'plus' },
  ];

  formData = {
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    cinPassport: '',
    country: '',
    birthDate: '',
    sex: '',
    phone: '',
    address: '',
    lastDiploma: '',
    diplomaYear: '',
    university: '',
    desiredDiploma: '',
    level: '',
    courseTime: '',
    knownBy: '',
  };

  countries = [
    'Afghanistan', 'Afrique du Sud', 'Albanie', 'Algérie', 'Allemagne', 'Andorre', 'Angola',
    'Antigua-et-Barbuda', 'Arabie saoudite', 'Argentine', 'Arménie', 'Australie', 'Autriche',
    'Azerbaïdjan', 'Bahamas', 'Bahreïn', 'Bangladesh', 'Barbade', 'Belgique', 'Belize', 'Bénin',
    'Bhoutan', 'Biélorussie', 'Birmanie', 'Bolivie', 'Bosnie-Herzégovine', 'Botswana', 'Brésil',
    'Brunei', 'Bulgarie', 'Burkina Faso', 'Burundi', 'Cambodge', 'Cameroun', 'Canada', 'Cap-Vert',
    'Chili', 'Chine', 'Chypre', 'Colombie', 'Comores', 'Congo', 'République démocratique du Congo',
    'Corée du Nord', 'Corée du Sud', 'Costa Rica', "Côte d'Ivoire", 'Croatie', 'Cuba', 'Danemark',
    'Djibouti', 'Dominique', 'Égypte', 'Émirats arabes unis', 'Équateur', 'Érythrée', 'Espagne',
    'Estonie', 'Eswatini', 'États-Unis', 'Éthiopie', 'Fidji', 'Finlande', 'France', 'Gabon',
    'Gambie', 'Géorgie', 'Ghana', 'Grèce', 'Grenade', 'Guatemala', 'Guinée', 'Guinée-Bissau',
    'Guinée équatoriale', 'Guyana', 'Haïti', 'Honduras', 'Hongrie', 'Îles Marshall', 'Îles Salomon',
    'Inde', 'Indonésie', 'Irak', 'Iran', 'Irlande', 'Islande', 'Israël', 'Italie', 'Jamaïque', 'Japon',
    'Jordanie', 'Kazakhstan', 'Kenya', 'Kirghizistan', 'Kiribati', 'Koweït', 'Laos', 'Lesotho',
    'Lettonie', 'Liban', 'Libéria', 'Libye', 'Liechtenstein', 'Lituanie', 'Luxembourg', 'Macédoine du Nord',
    'Madagascar', 'Malaisie', 'Malawi', 'Maldives', 'Mali', 'Malte', 'Maroc', 'Maurice', 'Mauritanie',
    'Mexique', 'Micronésie', 'Moldavie', 'Monaco', 'Mongolie', 'Monténégro', 'Mozambique', 'Namibie',
    'Nauru', 'Népal', 'Nicaragua', 'Niger', 'Nigeria', 'Norvège', 'Nouvelle-Zélande', 'Oman', 'Ouganda',
    'Ouzbékistan', 'Pakistan', 'Palaos', 'Panama', 'Papouasie-Nouvelle-Guinée', 'Paraguay', 'Pays-Bas',
    'Pérou', 'Philippines', 'Pologne', 'Portugal', 'Qatar', 'République centrafricaine',
    'République dominicaine', 'République tchèque', 'Roumanie', 'Royaume-Uni', 'Russie', 'Rwanda',
    'Saint-Christophe-et-Niévès', 'Saint-Marin', 'Saint-Vincent-et-les-Grenadines', 'Sainte-Lucie',
    'Salvador', 'Samoa', 'Sao Tomé-et-Principe', 'Sénégal', 'Serbie', 'Seychelles', 'Sierra Leone',
    'Singapour', 'Slovaquie', 'Slovénie', 'Somalie', 'Soudan', 'Soudan du Sud', 'Sri Lanka', 'Suède',
    'Suisse', 'Suriname', 'Syrie', 'Tadjikistan', 'Tanzanie', 'Tchad', 'Thaïlande', 'Timor oriental',
    'Togo', 'Tonga', 'Trinité-et-Tobago', 'Tunisie', 'Turkménistan', 'Turquie', 'Tuvalu', 'Ukraine',
    'Uruguay', 'Vanuatu', 'Vatican', 'Venezuela', 'Vietnam', 'Yémen', 'Zambie', 'Zimbabwe'
  ];
  diplomasByUniversity: Record<string, Array<{ type: 'header' | 'item'; text: string }>> = {
    ISB: [
      { type: 'header', text: 'Licences' },
      { type: 'item', text: 'Licence in Business Administration (ENG)' },
      { type: 'item', text: 'Licence en Business Intelligence (Bilingue)' },
      { type: 'item', text: 'Licence Nationale en Gestion (FR)' },
      { type: 'item', text: 'Licence Nationale en Comptabilité (FR)' },
      { type: 'item', text: 'Licence Nationale en Droit Privé (FR)' },
      { type: 'item', text: 'Licence Nationale en Droit des Affaires (FR)' },
      { type: 'item', text: 'Licence en science économique : Business Economics (with AI) (FR)' },
      { type: 'item', text: 'Licence en science économique : Transport et Logistique (FR)' },

      { type: 'header', text: 'Masters' },
      { type: 'item', text: 'Master in Business Administration (MBA) (ENG)' },
      { type: 'item', text: 'Mastère en Marketing Digital (FR)' },
      { type: 'item', text: 'Master en Comptabilité (FR)' },
      { type: 'item', text: 'Mastère Finance d’Entreprise et Ingénierie Financière (FR)' },
      { type: 'item', text: 'Master of Science (MSc) in Digital Marketing (ENG)' },
      { type: 'item', text: 'Master of Science (MSc) in Finance (ENG)' },
      { type: 'item', text: 'Master en Droit (FR)' },
    ],
    IIT: [
      { type: 'header', text: 'Préparatoire' },
      { type: 'item', text: 'Cycle Préparatoire' },

      { type: 'header', text: 'Licence' },
      { type: 'item', text: 'Licence en Informatique : Génie Logiciel et Système d’Information' },
      { type: 'item', text: 'Licence en Génie Industriel : Management des Systèmes Industriels' },

      { type: 'header', text: 'Masters' },
      { type: 'item', text: 'Master Industrie 4.0' },

      { type: 'header', text: 'Ingénierie' },
      { type: 'item', text: 'Génie Civil' },
      { type: 'item', text: 'Génie des Procédés' },
      { type: 'item', text: 'Génie Mécanique' },
      { type: 'item', text: 'Génie Industriel' },
      { type: 'item', text: 'Génie des Télécommunications' },
      { type: 'item', text: 'Génie Informatique' },

      { type: 'header', text: 'Architecture' },
      { type: 'item', text: 'Architecture' },
    ],
  };

  constructor(private authService: AuthService, private router: Router) { }

  get filteredCountries() {
    return this.countries.filter(c => c.toLowerCase().includes(this.countrySearch.toLowerCase()));
  }

  get filteredSexOptions() {
    return this.sexOptions.filter(s => s.toLowerCase().includes(this.sexSearch.toLowerCase()));
  }

  get progressPercentage() {
    return this.step() === 5 ? 100 : Math.round(((this.step() - 1) / 4) * 100);
  }
  onlyNumbers(event: any, fieldName: keyof typeof this.formData) {
    const input = event.target;
    input.value = input.value.replace(/[^0-9]/g, '');
    (this.formData as any)[fieldName] = input.value;
  }
  onUniversityChange() {
    this.formData.desiredDiploma = '';
  }

  validateStep(): boolean {
    const s = this.step();
    if (s === 1) {
      if (!this.formData.firstName || !this.formData.lastName || !this.formData.email || !this.formData.password) {
        this.error.set('Veuillez remplir tous les champs personnels.');
        return false;
      }
    } else if (s === 2) {
      if (!this.formData.cinPassport || !this.formData.country || !this.formData.birthDate || !this.formData.sex || !this.formData.phone || !this.formData.address) {
        this.error.set('Veuillez remplir toutes les informations personnelles.');
        return false;
      }
    } else if (s === 3) {
      if (!this.formData.lastDiploma || !this.formData.diplomaYear) {
        this.error.set('Veuillez remplir les informations sur votre diplôme obtenu.');
        return false;
      }
    } else if (s === 4) {
      if (!this.formData.university || !this.formData.desiredDiploma || !this.formData.level || !this.formData.courseTime || !this.formData.knownBy) {
        this.error.set('Veuillez remplir toutes les informations.');
        return false;
      }
    }
    return true;
  }

  handleNext() {
    if (this.validateStep()) {
      this.step.update(s => s + 1);
      this.error.set('');
    }
  }

  handleBack() {
    this.step.update(s => s - 1);
    this.error.set('');
  }

  selectCountry(country: string) {
    this.countrySearch = country;   // ⭐ met le pays dans l'input
    this.formData.country = country; // ⭐ sauvegarde le pays sélectionné
    this.showCountryDropdown.set(false); // ferme la liste
  }

  selectSex(sex: string) {
    this.sexSearch = sex;
    this.formData.sex = sex;
    this.showSexDropdown.set(false);
  }

  get filteredLastDiplomaOptions() {
    return this.diplomaOptions.filter(d => d.toLowerCase().includes(this.lastDiplomaSearch.toLowerCase()));
  }

  selectLastDiploma(diploma: string) {
    this.lastDiplomaSearch = diploma;
    this.formData.lastDiploma = diploma;
    this.showLastDiplomaDropdown.set(false);
  }

  get filteredUniversityOptions() {
    return this.universityOptions.filter(u => u.toLowerCase().includes(this.universitySearch.toLowerCase()));
  }

  selectUniversity(uni: string) {
    this.universitySearch = uni;
    this.formData.university = uni;
    this.showUniversityDropdown.set(false);
    this.onUniversityChange(); // Reset desired diploma
    this.desiredDiplomaSearch = ''; // Also reset search field for diploma
  }

  get filteredDesiredDiplomaOptions() {
    if (!this.formData.university) return [];
    const all = this.diplomasByUniversity[this.formData.university];
    const q = this.desiredDiplomaSearch.toLowerCase();
    if (!q) return all;
    // When searching: keep only matching items (hide headers)
    return all.filter(d => d.type === 'item' && d.text.toLowerCase().includes(q));
  }

  selectDesiredDiploma(diploma: string) {
    this.desiredDiplomaSearch = diploma;
    this.formData.desiredDiploma = diploma;
    this.showDesiredDiplomaDropdown.set(false);
  }

  get filteredLevelOptions() {
    return this.levelOptions.filter(l => l.includes(this.levelSearch));
  }

  selectLevel(level: string) {
    this.levelSearch = level;
    this.formData.level = level;
    this.showLevelDropdown.set(false);
  }

  get filteredCourseTimeOptions() {
    return this.courseTimeOptions.filter(c => c.toLowerCase().includes(this.courseTimeSearch.toLowerCase()));
  }

  selectCourseTime(course: string) {
    this.courseTimeSearch = course;
    this.formData.courseTime = course;
    this.showCourseTimeDropdown.set(false);
  }

  get filteredKnownByOptions() {
    return this.knownByOptions.filter(k =>
      k.label.toLowerCase().includes(this.knownBySearch.toLowerCase())
    );
  }

  selectKnownBy(knownBy: string) {
    this.knownBySearch = knownBy;
    this.formData.knownBy = knownBy;
    this.showKnownByDropdown.set(false);
  }

  async handleSubmit(e: Event) {
    e.preventDefault();
    if (!this.validateStep()) return;

    this.isSubmitting.set(true);
    try {
      await this.authService.register({
        nom: this.formData.lastName,
        prenom: this.formData.firstName,
        email: this.formData.email,
        motDePasse: this.formData.password,
        CIN: this.formData.cinPassport || undefined,
        pays: this.formData.country || undefined,
        dateNaissance: this.formData.birthDate || undefined,
        sexe: this.formData.sex || undefined,
        telephone: this.formData.phone || undefined,
        adresse: this.formData.address || undefined,
        dernierDiplomeObtenu: this.formData.lastDiploma || undefined,
        anneeObtention: this.formData.diplomaYear || undefined,
        universite: this.formData.university || undefined,
        diplomeSouhaite: this.formData.desiredDiploma || undefined,
        niveau: this.formData.level || undefined,
        cours: this.formData.courseTime || undefined,
        sourceConnaissanceUniversite: this.formData.knownBy || undefined,
      });
      this.step.set(5);
    } catch (err: any) {
      if (err?.status === 400) {
        this.error.set("Cet email est déjà utilisé.");
      } else {
        this.error.set("Une erreur est survenue lors de l'inscription.");
      }
    } finally {
      this.isSubmitting.set(false);
    }
  }

  goToDashboard() {
    this.router.navigate(['/student/dashboard']);
  }
}
