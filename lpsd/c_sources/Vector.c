
#include "Vector.h"

/*
 *
 *
 * Ingo Diepholz 08-06-2011
 *
 * $Id$
 */

/*****************************************************************************/
/***                                 PLUS                                  ***/
/*****************************************************************************/

Vector *vec_plus_s(Vector *v, double s) {
  plus_internal(v, s);
  return v;
}

Vector *s_plus_vec(double s, Vector *v) {
  plus_internal(v, s);
  return v;
}

Vector *vec_plus_vec(int num, Vector *v, ...) {

  int      i;
  Vector *v1 = v;
  va_list  args;
  va_start (args, v);

  /* get first and second vector */

  for (i=1; i<num; i++)
  {
    Vector *v2 = va_arg(args, Vector * );
    plus_internal_v(v1, v2);
  }

  va_end (args);
  return v1;
}

Vector *plus_internal_v(Vector *v1, Vector *v2) {

  long int ii;
  if (v1==NULL) {
    v1 = ALLOCATE(Vector);
  }
  if (v2==NULL) {
    return v1;
  }

  if ((v1->data == NULL) && (v2->data == NULL)) {
    /* nothing to do */
  }
  else if ((v1->data != NULL) && (v2->data == NULL)) {
    /* nothing to do */
  }
  else if ((v1->data == NULL) && (v2->data != NULL)) {
    /* Copy v2 to v1 */
    v1 = copyVector(v2);
  }
  else {
    for (ii = v1->size-1; ii >= 0; ii--) v1->data[ii] += v2->data[ii];
  }
  return v1;
}

Vector *plus_internal(Vector *v, double s) {

  long int ii;
  if (v==NULL) {
    v = ALLOCATE(Vector);
  }
  else {

    if (v->data == NULL) {
      /* nothing to do */
    }
    else {
      for (ii = v->size-1; ii >= 0; ii--) v->data[ii] += s;
    }
  }
  return v;
}



/*****************************************************************************/
/***                                MINUS                                  ***/
/*****************************************************************************/

Vector *vec_minus_s(Vector *v, double s) {
  plus_internal(v, -s);
  return v;
}

Vector *s_minus_vec(double s, Vector *v) {
  plus_internal(times_internal(v, -1), s);
  return v;
}

/*****************************************************************************/
/***                                TIMES                                  ***/
/*****************************************************************************/

Vector *vec_times_s(Vector *v, double s) {
  times_internal(v, s);
  return v;
}

Vector *s_times_vec(double s, Vector *v) {
  times_internal(v, s);
  return v;
}

Vector *vec_times_vec(int num, Vector *v, ...) {

  int      i;
  va_list  args;
  /* get first and second vector */
  Vector *v1 = v;
  va_start (args, v);

  for (i=1; i<num; i++)
  {
    Vector *v2 = va_arg(args, Vector * );
    times_internal_v(v1, v2);
  }

  va_end (args);
  return v1;
}

Vector *times_internal_v(Vector *v1, Vector *v2) {

  long int ii;
  if (v1==NULL) {
    v1 = ALLOCATE(Vector);
  }
  if (v2==NULL) {
    return v1;
  }

  if ((v1->data == NULL) && (v2->data == NULL)) {
    /* nothing to do */
  }
  else if ((v1->data != NULL) && (v2->data == NULL)) {
    /* nothing to do */
  }
  else if ((v1->data == NULL) && (v2->data != NULL)) {
    /* Copy v2 to v1 */
    v1 = copyVector(v2);
  }
  else {
    for (ii = v1->size-1; ii >= 0; ii--) v1->data[ii] *= v2->data[ii];
  }
  return v1;
}

Vector *times_internal(Vector *v, double s) {

  long int ii;
  if (v==NULL) {
    v = ALLOCATE(Vector);
  }
  else {

    if (v->data == NULL) {
      /* nothing to do */
    }
    else {
      for (ii = v->size-1; ii >= 0; ii--) {
        v->data[ii] *= s;
      }
    }
  }
  return v;
}

/*****************************************************************************/
/***                               DIVIDE                                  ***/
/*****************************************************************************/

Vector *vec_div_s(Vector *v, double s) {
  times_internal(v, 1/s);
  return v;
}


/*****************************************************************************/
/***                        Arithmetic Operators                           ***/
/*****************************************************************************/

Vector *sin_of_vec(Vector *v) {
  long int ii;
  if (v==NULL) {
    v = ALLOCATE(Vector);
  }
  else {

    if (v->data == NULL) {
      /* nothing to do */
    }
    else {
      for (ii = v->size-1; ii >= 0; ii--) {
        v->data[ii] = sin(v->data[ii]);
      }
    }
  }
  return v;
}

Vector *cos_of_vec(Vector *v) {
  long int ii;
  if (v==NULL) {
    v = ALLOCATE(Vector);
  }
  else {

    if (v->data == NULL) {
      /* nothing to do */
    }
    else {
      for (ii = v->size-1; ii >= 0; ii--) {
        v->data[ii] = cos(v->data[ii]);
      }
    }
  }
  return v;
}



/*****************************************************************************/
/***                                 MISC                                  ***/
/*****************************************************************************/

Vector *copyVector(Vector *in) {

  Vector *out  = ALLOCATE(Vector);
  double *dout = ALLOCATE_ARRAY(double, in->size);
  double *din  = in->data;
  long int i;

  out->data = dout;
  out->size = in->size;

  for ( i = out->size-1; i >= 0; i-- )
    *dout++ = *din++;

  return out;
}

Vector *initVector(double *data, long int size) {
  Vector *v = ALLOCATE(Vector);
  v->data = data;
  v->size = size;
  return v;
}

char *toString(Vector *t) {

  static char retbuf[300] = "";
  long int ll;

  if (t==NULL) {
    return "NULL";
  }

  retbuf[0] = '\0';
  strcpy(retbuf, "[");
  for(ll=0; ll<myMin(10,t->size); ll++) {
    sprintf(retbuf, "%s%16.15e, ", retbuf, t->data[ll]);
  }
  /* Remove last blank */
  retbuf[strlen(retbuf)-2] = ']';
  retbuf[strlen(retbuf)-1] = '\0';

  return retbuf;
}

int myMin(long int a, long int b) {
  return ((a<b) ? a : b);
}

void freeVector(Vector *v) {
  if (v != NULL) {
    if (v->data && v->data != NULL)
      free(v->data);
    free(v);
    printMsg(DEBUG, "----------- Free !!!\n");
  }
}

void printMsg(int level, char *format, ...) {
  char buffer[400];
  char lvlMsg[10];
  va_list  args;
  va_start (args, format);
  vsprintf (buffer,format, args);

  if ((level!=NONE) && (VERBOSE_LEVEL>=level)) {
    switch(level){
      case DEBUG:   strcpy(lvlMsg, "DEBUG");   break;
      case INFO:    strcpy(lvlMsg, "INFO");    break;
      case WARNING: strcpy(lvlMsg, "WARNING"); break;
    }

    printf("%-9s %s\n", lvlMsg, buffer);
  }
  va_end (args);
}
