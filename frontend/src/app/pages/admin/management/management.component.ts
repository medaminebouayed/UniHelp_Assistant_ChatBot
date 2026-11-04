import { Component, signal, computed, inject, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';
import { animate, style, transition, trigger, state } from '@angular/animations';
import { AdminService, BackendAdmin } from '../../../services/admin.service';

interface Admin {
  id: number;
  name: string;
  prenom: string;
  email: string;
  password: string;
  phone: string;
  role: string;
  department: string;
  status: string;
  lastLogin: string;
}

@Component({
  selector: 'app-admin-management',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule, TranslateModule],
  templateUrl: './management.component.html',
  animations: [
    trigger('modalSlide', [
      transition(':enter', [
        style({ transform: 'scale(0.95) translateY(20px)', opacity: 0 }),
        animate('300ms cubic-bezier(0.34, 1.56, 0.64, 1)', style({ transform: 'scale(1) translateY(0)', opacity: 1 }))
      ]),
      transition(':leave', [
        animate('200ms ease-in', style({ transform: 'scale(0.95) translateY(20px)', opacity: 0 }))
      ])
    ])
  ]
})
export class ManagementComponent implements OnInit {
  private adminService = inject(AdminService);

  admins = signal<Admin[]>([]);
  loading = signal(true);

  ngOnInit(): void {
    this.loadAdmins();
  }

  private loadAdmins(): void {
    this.adminService.getAll().subscribe({
      next: (list) => {
        this.admins.set(
          list.map((a: any) => ({
            id: a.id,
            name: a.nom,
            prenom: a.prenom,
            email: a.email,
            password: '••••••••',
            phone: a.telephone || '-',
            role: 'Admin',
            department: a.departement || 'Non défini',
            status: 'Actif',
            lastLogin: '-',
          }))
        );
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }

  searchQuery = signal('');
  isModalOpen = signal(false);
  isDeleteModalOpen = signal(false);
  viewingAdmin = signal<Admin | null>(null);

  formData = { name: '', prenom: '', email: '', password: '', phone: '', department: 'Scolarité' };
  departments = ['Scolarité', 'Examens', 'Direction', 'Ressources Humaines'];

  filteredAdmins = computed(() => {
    const q = this.searchQuery().toLowerCase();
    return this.admins().filter(a => a.name.toLowerCase().includes(q) || a.email.toLowerCase().includes(q));
  });

  currentPage = signal(1);
  itemsPerPage = 5;

  filteredCount = computed(() => this.filteredAdmins().length);
  totalPages = computed(() => Math.ceil(this.filteredCount() / this.itemsPerPage));
  startIndex = computed(() => (this.currentPage() - 1) * this.itemsPerPage);
  endIndex = computed(() => Math.min(this.startIndex() + this.itemsPerPage, this.filteredCount()));

  paginatedAdmins = computed(() => {
    const start = this.startIndex();
    return this.filteredAdmins().slice(start, start + this.itemsPerPage);
  });

  pageNumbers = computed(() => {
    const total = this.totalPages();
    const current = this.currentPage();
    let start = Math.max(1, current - 2);
    let end = Math.min(total, start + 4);
    if (end - start < 4) {
      start = Math.max(1, end - 4);
    }
    return Array.from({ length: Math.max(0, end - start + 1) }, (_, i) => start + i);
  });

  openAddModal() {
    this.viewingAdmin.set(null);
    this.formData = { name: '', prenom: '', email: '', password: '', phone: '', department: 'Scolarité' };
    this.isModalOpen.set(true);
  }

  openViewModal(admin: Admin) {
    this.viewingAdmin.set(admin);
    this.isModalOpen.set(true);
  }

  openDeleteModal(admin: Admin) {
    this.viewingAdmin.set(admin);
    this.isDeleteModalOpen.set(true);
  }

  handleSave() {
    if (this.formData.name && this.formData.email) {
      this.adminService
        .create({
          nom: this.formData.name,
          prenom: this.formData.prenom,
          email: this.formData.email,
          motDePasse: this.formData.password,
          telephone: this.formData.phone || undefined,
          departement: this.formData.department,
        })
        .subscribe({
          next: () => {
            this.isModalOpen.set(false);
            this.loadAdmins();
          },
          error: () => {
            this.isModalOpen.set(false);
            this.loadAdmins();
          },
        });
    }
  }

  handleDelete() {
    const admin = this.viewingAdmin();
    if (admin) {
      this.adminService.delete(admin.id).subscribe({
        next: () => {
          this.isDeleteModalOpen.set(false);
          this.loadAdmins();
        },
        error: () => {
          this.isDeleteModalOpen.set(false);
          this.loadAdmins();
        },
      });
    }
  }

  prevPage() { if (this.currentPage() > 1) this.currentPage.update(v => v - 1); }
  nextPage() { if (this.currentPage() < this.totalPages()) this.currentPage.update(v => v + 1); }
  goToPage(p: number) { this.currentPage.set(p); }
}
