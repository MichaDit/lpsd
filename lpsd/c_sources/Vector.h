/*
 * Header for Vector.c
 *
 * $Id$
 */

#ifndef VECTOR_H
#define VECTOR_H

#ifndef ALLOCATE_ARRAY
#define ALLOCATE_ARRAY(type, n) ((type *)calloc(n, sizeof(type)))
#endif

#ifndef ALLOCATE
#define ALLOCATE(type)          ((type *)calloc(1, sizeof(type)))
#endif

#define NONE          -1
#define WARNING        1
#define INFO           2
#define DEBUG          3
#define VERBOSE_LEVEL  INFO

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Vector
{
  /* size of vector */
  long int size;

  /* data array containing vector elements */
  double *data;
} Vector;


/* from Vector.c */
void    printMsg(int level, char *format, ...);
char   *toString(Vector *t);
Vector *initVector(double *data, long int size);
Vector *copyVector(Vector *in);
void    freeVector(Vector *v);


/* vector operations */
Vector *vec_plus_s(Vector *v, double s);
Vector *s_plus_vec(double s, Vector *v);
Vector *vec_plus_vec(int num, Vector *v, ...);
Vector *plus_internal(Vector *v, double s);
Vector *plus_internal_v(Vector *v1, Vector *v2);

Vector *vec_minus_s(Vector *v, double s);
Vector *s_minus_vec(double s, Vector *v);

Vector *vec_times_s(Vector *v, double s);
Vector *s_times_vec(double s, Vector *v);
Vector *vec_times_vec(int num, Vector *v, ...);
Vector *times_internal(Vector *v, double s);
Vector *times_internal_v(Vector *v1, Vector *v2);

Vector *vec_div_s(Vector *v, double s);

Vector *sin_of_vec(Vector *v);
Vector *cos_of_vec(Vector *v);

#endif /* VECTOR_H */
