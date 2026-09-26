import * as cdk from 'aws-cdk-lib/core';
import { Match, Template } from 'aws-cdk-lib/assertions';
import { CORPUS_PREFIX, GrepRagStack, RUNTIME_LOG_GROUP_NAME, SESSION_MOUNT } from '../lib/grep_rag-stack';

test('runtime mounts session storage and can read the corpus bucket', () => {
  const app = new cdk.App();
  const stack = new GrepRagStack(app, 'GrepRagStack', {
    env: { account: '123456789012', region: 'us-east-1' },
  });
  const template = Template.fromStack(stack);

  template.resourceCountIs('AWS::BedrockAgentCore::Runtime', 1);
  template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
    AgentRuntimeName: 'grep_rag',
    FilesystemConfigurations: [
      { SessionStorage: { MountPath: SESSION_MOUNT } },
    ],
    EnvironmentVariables: Match.objectLike({
      CORPUS_PREFIX,
      SESSION_MOUNT,
      MODEL_ID: 'us.moonshotai.kimi-k3',
      AWS_REGION: 'us-east-1',
    }),
  });

  template.hasResourceProperties('AWS::Logs::LogGroup', {
    LogGroupName: RUNTIME_LOG_GROUP_NAME,
  });
  template.resourceCountIs('AWS::Logs::Delivery', 2);

  template.resourceCountIs('Custom::CDKBucketDeployment', 1);
  template.hasResourceProperties('Custom::CDKBucketDeployment', {
    DestinationBucketKeyPrefix: CORPUS_PREFIX,
    Prune: true,
  });

  template.hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Action: Match.arrayWith([
            's3:GetObject*',
            's3:GetBucket*',
            's3:List*',
          ]),
          Effect: 'Allow',
        }),
        Match.objectLike({
          Action: Match.arrayWith([
            'bedrock:InvokeModel',
            'bedrock:InvokeModelWithResponseStream',
          ]),
          Effect: 'Allow',
        }),
      ]),
    },
  });
});
