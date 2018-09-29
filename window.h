#ifndef WINDOW_H
#define WINDOW_H

#include <QOpenGLWindow>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

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
    QOpenGLTexture           *m_texture;

    void printInfo();
};

#endif // WINDOW_H
