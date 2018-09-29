#ifndef WINDOW_H
#define WINDOW_H

#include <QOpenGLWindow>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>

class QOpenGLShaderProgram;

class Window : public QOpenGLWindow,
               protected QOpenGLFunctions
{
    Q_OBJECT

public:
    Window();
    ~Window();

    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();
    void teardownGL();
protected slots:
    void update();

protected:
  void keyPressEvent(QKeyEvent *event);
  void keyReleaseEvent(QKeyEvent *event);
  void mousePressEvent(QMouseEvent *event);
  void mouseReleaseEvent(QMouseEvent *event);

private:
    // OpenGL State Information
    QOpenGLBuffer             m_vertex;
    QOpenGLVertexArrayObject  m_object;
    QOpenGLShaderProgram     *m_program;

    void printInfo();
};

#endif // WINDOW_H
