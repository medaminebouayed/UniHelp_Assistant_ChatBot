import { inject } from '@angular/core';
import { Router, CanActivateFn, ActivatedRouteSnapshot } from '@angular/router';
import { AuthService } from '../services/auth.service';

export const authGuard: CanActivateFn = (route: ActivatedRouteSnapshot) => {
  const authService = inject(AuthService);
  const router = inject(Router);

  const token = localStorage.getItem('access_token');
  const user = authService.user();

  if (!token && !user) {
    return router.parseUrl('/login');
  }

  // If token exists but user not loaded yet, allow navigation (loading state will show)
  const allowedRole = route.data['role'] as 'student' | 'admin' | undefined;

  if (user && allowedRole && user.role !== allowedRole) {
    const redirectPath = user.role === 'admin' ? '/admin/dashboard' : '/student/dashboard';
    return router.parseUrl(redirectPath);
  }

  return true;
};
