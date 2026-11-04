import { Routes } from '@angular/router';
import { AuthLayoutComponent } from './layouts/auth-layout/auth-layout.component';
import { StudentLayoutComponent } from './layouts/student-layout/student-layout.component';
import { AdminLayoutComponent } from './layouts/admin-layout/admin-layout.component';
import { LoginComponent } from './pages/auth/login/login.component';
import { authGuard } from './guards/auth.guard';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    redirectTo: 'login'
  },
  {
    path: '',
    component: AuthLayoutComponent,
    children: [
      {
        path: 'login',
        component: LoginComponent
      }
    ]
  },
  {
    path: 'register',
    loadComponent: () => import('./pages/auth/register/register.component').then(m => m.RegisterComponent)
  },
  {
    path: 'student',
    component: StudentLayoutComponent,
    canActivate: [authGuard],
    data: { role: 'student' },
    children: [
      {
        path: 'dashboard',
        loadComponent: () => import('./pages/student/dashboard/dashboard.component').then(m => m.DashboardComponent)
      },
      {
        path: 'profile',
        loadComponent: () => import('./pages/student/profile/profile.component').then(m => m.ProfileComponent)
      },
      {
        path: 'emails',
        loadComponent: () => import('./pages/student/emails/email-generator.component').then(m => m.EmailGeneratorComponent)
      },
      {
        path: 'faq',
        loadComponent: () => import('./pages/student/faq/faq.component').then(m => m.FaqComponent)
      },
      {
        path: 'chatbot',
        loadComponent: () => import('./pages/student/chatbot/chatbot.component').then(m => m.ChatbotComponent)
      },
      {
        path: 'settings',
        loadComponent: () => import('./pages/student/settings/settings.component').then(m => m.SettingsComponent)
      },
      {
        path: '',
        pathMatch: 'full',
        redirectTo: 'dashboard'
      }
    ]
  },
  {
    path: 'admin',
    component: AdminLayoutComponent,
    canActivate: [authGuard],
    data: { role: 'admin' },
    children: [
      {
        path: 'dashboard',
        loadComponent: () => import('./pages/admin/dashboard/admin-dashboard.component').then(m => m.AdminDashboardComponent)
      },
      {
        path: 'students',
        loadComponent: () => import('./pages/admin/students/students-list.component').then(m => m.StudentsListComponent)
      },
      {
        path: 'documents',
        loadComponent: () => import('./pages/admin/documents/documents.component').then(m => m.DocumentsComponent)
      },
      {
        path: 'management',
        loadComponent: () => import('./pages/admin/management/management.component').then(m => m.ManagementComponent)
      },
      {
        path: 'profile',
        loadComponent: () => import('./pages/admin/profile/profile.component').then(m => m.AdminProfileComponent)
      },
      {
        path: 'settings',
        loadComponent: () => import('./pages/admin/settings/settings.component').then(m => m.AdminSettingsComponent)
      },
      {
        path: '',
        pathMatch: 'full',
        redirectTo: 'dashboard'
      }
    ]
  },
  {
    path: '**',
    redirectTo: 'login'
  }
];
