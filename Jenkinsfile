pipeline{
    agent {label "dev" }

    stages{

        // 1. clone code 
        // 1.1 sonarqube
        // 1.3 installing dependencies
        // 1.4 owasp dependency check 
        // 2. trivy file scan
        // 2.1 building images 
        // 2.2 push images 
        // 2.3 docker image scan  
        // 4. restart deployed 
        

        stage("Clone Code"){
           steps{
            git url : "https://github.com/Saroj-kr-tharu/GlamAI---model", branch :"main"
         } }
        
        stage('SonarQube Analysis') {
            steps {
                script {
                    def scannerHome = tool 'SonarQube Scanner'
                    withSonarQubeEnv('SonarQube') {
                        sh """
                            ${scannerHome}/bin/sonar-scanner \
                            -Dsonar.projectKey=cms \
                            -Dsonar.sources=.
                        """
                    }
                }
            }
        }
        
        stage('Quality Gate') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    waitForQualityGate abortPipeline: true
                }
            }
        }

        stage("OWASP Dependency Check"){
          steps{
              script {
                 
                  sh 'mkdir -p dependency-check-report'
                  
                  def dependencyCheckHome = tool 'OWASP Dependency-Check'
                  
                 
                  withCredentials([string(credentialsId: 'NVD_API_KEY', variable: 'NVD_API_KEY')]) {
                      sh """
                          ${dependencyCheckHome}/bin/dependency-check.sh \
                          --scan . \
                          --format XML \
                          --out dependency-check-report \
                          --prettyPrint \
                          --nvdApiKey \${NVD_API_KEY}
                      """
                  }
              }
              
              // Publish the report
              dependencyCheckPublisher pattern: 'dependency-check-report/dependency-check-report.xml'
          }
      }


        stage("scan file system"){ steps{ 
            sh 'trivy fs . -o result.json'
         } }

        stage(" Build Docker Image "){ 
          steps{
             echo " Buildinig Docker Image  "

             withCredentials(  [usernamePassword(
                        credentialsId: "dockerHubCreds",
                        passwordVariable:"dockerHubPass" ,
                        usernameVariable:"dockerHubUser" )]
                    ){
                        sh '''
                          docker build -t ${dockerHubUser}/glamai-model:latest .
                        '''
                       
                     }
          

              
            } 
         }

        stage("Docker Image Scan") {
          steps {
              withCredentials([usernamePassword(
                  credentialsId: "dockerHubCreds",
                  passwordVariable: "dockerHubPass",
                  usernameVariable: "dockerHubUser"
              )]) {
                  sh '''
                    trivy image --format json -o model-image-scan.json ${dockerHubUser}/glamai-model:latest
                    
                  '''
              }
          }
      }

        stage("Image Push To DockerHub "){ 
            steps{
                echo "Image Pushing to DockerHub "
                
                withCredentials(  [usernamePassword(
                        credentialsId: "dockerHubCreds",
                        passwordVariable:"dockerHubPass" ,
                        usernameVariable:"dockerHubUser" )]
                    ){
                        sh '''
                          echo "${dockerHubPass}" | docker login -u "${dockerHubUser}" --password-stdin
                          docker push ${dockerHubUser}/glamai-model:latest
                          
                        '''
                       
                     }
            } 
         }

        stage("k8s Deployment Restart"){ 
          steps{
             echo " k8s deploymnet restarting  "
             sh "kubectl rollout restart deployment model-dep -n cms-ns"
         } }
    }


   

  post {
    success {
        script {
            buildUserVars()
            def user = env.BUILD_USER ?: 'System / Webhook'
            emailext(
                mimeType: 'text/html',
                attachmentsPattern: 'result.json, model-image-scan.json,  dependency-check-report/dependency-check-report.xml',
                from: 'sarojtestingkrtharu@gmail.com',
                to: 'sarojc11345@gmail.com',
                subject: "✅ Build Success – ${env.JOB_NAME} ",
                body: """
                <html>
                <body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;background-color:#f4f6f8;">
                  <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                      <td align="center" style="padding:30px;">
                        <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.1);overflow:hidden;">
                          
                          <tr>
                            <td style="background:#22c55e;color:#ffffff;padding:20px;text-align:center;">
                              <h1 style="margin:0;font-size:24px;">🎉 Build Successful</h1>
                            </td>
                          </tr>

                          <tr>
                            <td style="padding:25px;color:#333333;">
                              <p style="font-size:16px;">Your Jenkins build completed successfully.</p>

                              <table width="100%" style="margin-top:15px;font-size:14px;">
                                <tr>
                                  <td><strong>Project</strong></td>
                                  <td>${env.JOB_NAME}</td>
                                </tr>
                                <tr>
                                  <td><strong>Build Number</strong></td>
                                  <td>#${env.BUILD_NUMBER}</td>
                                </tr>
                                <tr>
                                  <td><strong>Triggered By </strong></td>
                                  <td>#${user}</td>
                                </tr>
                                <tr>
                                  <td><strong>Status</strong></td>
                                  <td style="color:#22c55e;font-weight:bold;">${currentBuild.currentResult}</td>
                                </tr>
                              </table>

                              <div style="margin-top:25px;text-align:center;">
                                <a href="${env.BUILD_URL}"
                                   style="background:#22c55e;color:#ffffff;text-decoration:none;
                                          padding:12px 22px;border-radius:6px;
                                          display:inline-block;font-weight:bold;">
                                  View Build Details
                                </a>
                              </div>
                            </td>
                          </tr>

                          <tr>
                            <td style="background:#f1f5f9;color:#6b7280;
                                       text-align:center;padding:15px;font-size:12px;">
                              Jenkins CI/CD • ${env.JOB_NAME}
                            </td>
                          </tr>

                        </table>
                      </td>
                    </tr>
                  </table>
                </body>
                </html>
                """
            )
        }
    }

    failure {
        script {
            buildUserVars()
            def user = env.BUILD_USER ?: 'System / Webhook'
            emailext(
                mimeType: 'text/html',
                attachmentsPattern: 'result.json, backend-image-scan.json, frontend-image-scan.json, dependency-check-report/dependency-check-report.xml',
                from: 'sarojtestingkrtharu@gmail.com',
                to: 'sarojc11345@gmail.com',
                subject: '❌ Build Failed – Todo App',
                body: """
                <html>
                <body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;background-color:#f4f6f8;">
                  <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                      <td align="center" style="padding:30px;">
                        <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.1);overflow:hidden;">
                          
                          <tr>
                            <td style="background:#ef4444;color:#ffffff;padding:20px;text-align:center;">
                              <h1 style="margin:0;font-size:24px;">🚨 Build Failed</h1>
                            </td>
                          </tr>

                          <tr>
                            <td style="padding:25px;color:#333333;">
                              <p style="font-size:16px;">The Jenkins build has failed. Please review the details below.</p>

                              <table width="100%" style="margin-top:15px;font-size:14px;">
                                <tr>
                                  <td><strong>Project</strong></td>
                                  <td>${env.JOB_NAME}</td>
                                </tr>
                                <tr>
                                  <td><strong>Build Number</strong></td>
                                  <td>#${env.BUILD_NUMBER}</td>
                                </tr>
                                <tr>
                                  <td><strong>Triggered By</strong></td>
                                  <td>#${user}</td>
                                </tr>
                                <tr>
                                  <td><strong>Status</strong></td>
                                  <td style="color:#ef4444;font-weight:bold;">${currentBuild.currentResult}</td>
                                </tr>
                              </table>

                              <div style="margin-top:25px;text-align:center;">
                                <a href="${env.BUILD_URL}"
                                   style="background:#ef4444;color:#ffffff;text-decoration:none;
                                          padding:12px 22px;border-radius:6px;
                                          display:inline-block;font-weight:bold;">
                                  View Failure Logs
                                </a>
                              </div>
                            </td>
                          </tr>

                          <tr>
                            <td style="background:#f1f5f9;color:#6b7280;
                                       text-align:center;padding:15px;font-size:12px;">
                              Jenkins CI/CD • Todo App
                            </td>
                          </tr>

                        </table>
                      </td>
                    </tr>
                  </table>
                </body>
                </html>
                """
            )
        }
    }
    }



}